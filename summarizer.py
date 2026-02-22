import argparse
import inspect
from dataclasses import dataclass

import torch
import evaluate
import nltk
import numpy as np
from datasets import DatasetDict, load_dataset
from transformers import (
    AutoModelForSeq2SeqLM,
    AutoTokenizer,
    DataCollatorForSeq2Seq,
    Seq2SeqTrainer,
    Seq2SeqTrainingArguments,
)

# Workaround for some Windows Python environments where importing torch after
# datasets/evaluate can trigger WinError 1114 for c10.dll.
_ = torch.__version__


MODEL_CATALOG = {
    "t5": "google-t5/t5-small",
    "bart": "facebook/bart-base",
    "pegasus": "google/pegasus-xsum",
}

DATASET_CONFIG = {
    "cnn_dailymail": {"dataset_args": ("cnn_dailymail", "3.0.0"), "text_col": "article", "summary_col": "highlights"},
    "xsum": {"dataset_args": ("xsum",), "text_col": "document", "summary_col": "summary"},
}


@dataclass
class PipelineConfig:
    model_key: str
    dataset_name: str
    output_dir: str
    max_input_length: int = 512
    max_target_length: int = 128
    train_samples: int | None = 20000
    val_samples: int | None = 2000
    test_samples: int | None = 2000


def get_dataset(config: PipelineConfig) -> tuple[DatasetDict, str, str]:
    ds_conf = DATASET_CONFIG[config.dataset_name]
    dataset = load_dataset(*ds_conf["dataset_args"])

    if config.train_samples:
        dataset["train"] = dataset["train"].select(range(min(config.train_samples, len(dataset["train"]))))
    if config.val_samples:
        dataset["validation"] = dataset["validation"].select(range(min(config.val_samples, len(dataset["validation"]))))
    if config.test_samples:
        dataset["test"] = dataset["test"].select(range(min(config.test_samples, len(dataset["test"]))))

    return dataset, ds_conf["text_col"], ds_conf["summary_col"]


def build_preprocess(tokenizer, text_col: str, summary_col: str, max_input_length: int, max_target_length: int):
    def preprocess_function(examples):
        inputs = examples[text_col]
        if tokenizer.name_or_path.startswith("google-t5/"):
            inputs = [f"summarize: {text}" for text in inputs]

        model_inputs = tokenizer(inputs, max_length=max_input_length, truncation=True)
        labels = tokenizer(text_target=examples[summary_col], max_length=max_target_length, truncation=True)
        model_inputs["labels"] = labels["input_ids"]
        return model_inputs

    return preprocess_function


def should_use_t5_prefix(tokenizer, model=None, model_path: str | None = None) -> bool:
    name = (getattr(tokenizer, "name_or_path", "") or "").lower()
    model_type = getattr(getattr(model, "config", None), "model_type", "")
    model_type = model_type.lower() if isinstance(model_type, str) else ""
    path_text = (model_path or "").lower()
    return "t5" in name or model_type == "t5" or "t5" in path_text


def build_compute_metrics(tokenizer):
    rouge_metric = evaluate.load("rouge")
    bleu_metric = evaluate.load("bleu")

    def postprocess_text(preds, labels):
        preds = [pred.strip() for pred in preds]
        labels = [label.strip() for label in labels]
        try:
            preds = ["\n".join(nltk.sent_tokenize(pred)) for pred in preds]
            labels = ["\n".join(nltk.sent_tokenize(label)) for label in labels]
        except LookupError:
            # Fallback if punkt resources are unavailable in the runtime.
            preds = [pred if pred else "" for pred in preds]
            labels = [label if label else "" for label in labels]
        return preds, labels

    def compute_metrics(eval_preds):
        preds, labels = eval_preds
        if isinstance(preds, tuple):
            preds = preds[0]
        preds = np.asarray(preds)
        labels = np.asarray(labels)

        # Newer transformers versions can return logits/float arrays here.
        if preds.ndim == 3:
            preds = np.argmax(preds, axis=-1)

        vocab_max = len(tokenizer) - 1
        preds = np.where(preds < 0, tokenizer.pad_token_id, preds)
        preds = np.clip(preds, 0, vocab_max).astype(np.int64)

        labels = np.where(labels != -100, labels, tokenizer.pad_token_id)
        labels = np.clip(labels, 0, vocab_max).astype(np.int64)

        decoded_preds = tokenizer.batch_decode(preds.tolist(), skip_special_tokens=True)
        decoded_labels = tokenizer.batch_decode(labels.tolist(), skip_special_tokens=True)

        decoded_preds, decoded_labels = postprocess_text(decoded_preds, decoded_labels)
        rouge_result = rouge_metric.compute(predictions=decoded_preds, references=decoded_labels, use_stemmer=True)

        try:
            bleu_result = bleu_metric.compute(
                predictions=decoded_preds,
                references=[[ref] for ref in decoded_labels],
            )
            bleu_score = round(float(bleu_result["bleu"]), 4)
        except Exception:
            bleu_score = 0.0

        result = {
            "rouge1": round(rouge_result["rouge1"], 4),
            "rouge2": round(rouge_result["rouge2"], 4),
            "rougeL": round(rouge_result["rougeL"], 4),
            "rougeLsum": round(rouge_result["rougeLsum"], 4),
            "bleu": bleu_score,
        }
        prediction_lens = [np.count_nonzero(pred != tokenizer.pad_token_id) for pred in preds]
        result["gen_len"] = round(float(np.mean(prediction_lens)), 2)
        return result

    return compute_metrics


def train_and_evaluate(config: PipelineConfig):
    nltk.download("punkt", quiet=True)
    nltk.download("punkt_tab", quiet=True)
    model_name = MODEL_CATALOG[config.model_key]
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForSeq2SeqLM.from_pretrained(model_name)

    dataset, text_col, summary_col = get_dataset(config)
    preprocess = build_preprocess(
        tokenizer=tokenizer,
        text_col=text_col,
        summary_col=summary_col,
        max_input_length=config.max_input_length,
        max_target_length=config.max_target_length,
    )
    tokenized_dataset = dataset.map(preprocess, batched=True, remove_columns=dataset["train"].column_names)

    data_collator = DataCollatorForSeq2Seq(tokenizer=tokenizer, model=model)
    training_kwargs = {
        "output_dir": config.output_dir,
        "save_strategy": "epoch",
        "logging_strategy": "steps",
        "logging_steps": 50,
        "learning_rate": 2e-5,
        "per_device_train_batch_size": 4,
        "per_device_eval_batch_size": 4,
        "weight_decay": 0.01,
        "save_total_limit": 2,
        "num_train_epochs": 1,
        "predict_with_generate": True,
        "generation_max_length": config.max_target_length,
        "fp16": False,
        "push_to_hub": False,
        "report_to": "none",
    }
    ta_params = inspect.signature(Seq2SeqTrainingArguments.__init__).parameters
    if "eval_strategy" in ta_params:
        training_kwargs["eval_strategy"] = "epoch"
    else:
        training_kwargs["evaluation_strategy"] = "epoch"
    training_args = Seq2SeqTrainingArguments(**training_kwargs)

    trainer_kwargs = {
        "model": model,
        "args": training_args,
        "train_dataset": tokenized_dataset["train"],
        "eval_dataset": tokenized_dataset["validation"],
        "data_collator": data_collator,
        "compute_metrics": build_compute_metrics(tokenizer),
    }
    trainer_params = inspect.signature(Seq2SeqTrainer.__init__).parameters
    if "processing_class" in trainer_params:
        trainer_kwargs["processing_class"] = tokenizer
    else:
        trainer_kwargs["tokenizer"] = tokenizer
    trainer = Seq2SeqTrainer(**trainer_kwargs)

    trainer.train()
    eval_metrics = trainer.evaluate(eval_dataset=tokenized_dataset["test"], metric_key_prefix="test")

    trainer.save_model(config.output_dir)
    tokenizer.save_pretrained(config.output_dir)
    print("Final test metrics:", eval_metrics)


def summarize(model_path: str, text: str, max_input_length: int = 512, max_output_length: int = 128) -> str:
    tokenizer = AutoTokenizer.from_pretrained(model_path)
    model = AutoModelForSeq2SeqLM.from_pretrained(model_path)

    prompt = text.strip()
    if should_use_t5_prefix(tokenizer=tokenizer, model=model, model_path=model_path):
        prompt = f"summarize: {prompt}"

    inputs = tokenizer(prompt, return_tensors="pt", truncation=True, max_length=max_input_length)
    summary_ids = model.generate(
        **inputs,
        max_length=max_output_length,
        min_length=30,
        num_beams=4,
        length_penalty=2.0,
        early_stopping=True,
    )
    return tokenizer.decode(summary_ids[0], skip_special_tokens=True)


def parse_args():
    parser = argparse.ArgumentParser(description="Fine-tune and evaluate transformer models for abstractive summarization.")
    parser.add_argument("--mode", choices=["train", "summarize"], default="train")
    parser.add_argument("--model", choices=list(MODEL_CATALOG.keys()), default="bart", help="Model family to fine-tune.")
    parser.add_argument("--dataset", choices=list(DATASET_CONFIG.keys()), default="cnn_dailymail")
    parser.add_argument("--output_dir", default="./finetuned_summarizer")
    parser.add_argument("--max_input_length", type=int, default=512)
    parser.add_argument("--max_target_length", type=int, default=128)
    parser.add_argument("--train_samples", type=int, default=20000)
    parser.add_argument("--val_samples", type=int, default=2000)
    parser.add_argument("--test_samples", type=int, default=2000)
    parser.add_argument("--model_path", default="./finetuned_summarizer")
    parser.add_argument("--text", default="")
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    if args.mode == "train":
        cfg = PipelineConfig(
            model_key=args.model,
            dataset_name=args.dataset,
            output_dir=args.output_dir,
            max_input_length=args.max_input_length,
            max_target_length=args.max_target_length,
            train_samples=args.train_samples,
            val_samples=args.val_samples,
            test_samples=args.test_samples,
        )
        train_and_evaluate(cfg)
    else:
        if not args.text.strip():
            raise ValueError("Provide text with --text when using --mode summarize.")
        print(summarize(model_path=args.model_path, text=args.text, max_input_length=args.max_input_length, max_output_length=args.max_target_length))
