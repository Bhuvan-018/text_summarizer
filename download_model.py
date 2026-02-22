# Dataset loading using Hugging Face Datasets
from datasets import load_dataset

def load_cnn_daily_mail():
    dataset = load_dataset("cnn_dailymail", "3.0.0")
    return dataset

def load_xsum():
    dataset = load_dataset("xsum")
    return dataset

# Example usage
if __name__ == "__main__":
    cnn_dm = load_cnn_daily_mail()
    xsum = load_xsum()
    print("CNN/Daily Mail sample:", cnn_dm["train"][0])
    print("XSUM sample:", xsum["train"][0])
from huggingface_hub import snapshot_download
import os

print("Starting download...")
try:
    path = snapshot_download(repo_id="google-t5/t5-small", cache_dir="./manual_cache", local_dir="./manual_model")
    print(f"Downloaded to {path}")
    print("Files in directory:")
    for f in os.listdir(path):
        print(f)
except Exception as e:
    print(f"Error: {e}")
