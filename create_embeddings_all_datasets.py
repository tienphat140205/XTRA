import os
import numpy as np
import torch
from transformers import AutoModel, AutoTokenizer
from pathlib import Path

# Cấu hình
MODEL_NAME = "BAAI/bge-m3"
BATCH_SIZE = 32
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

# Gốc của dự án
ROOT_DIR = os.path.abspath(os.getcwd())  # This will use the current directory (XTRA)
def load_model_and_tokenizer(model_name=MODEL_NAME):
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModel.from_pretrained(model_name)
    return tokenizer, model


def create_doc_embeddings(text_file, output_file, tokenizer, model, device=DEVICE, batch_size=BATCH_SIZE):
    if not os.path.exists(text_file):
        print(f"Text file {text_file} not found, skipping.")
        return
    with open(text_file, "r", encoding="utf-8") as f:
        texts = [line.strip() for line in f]
    model.eval()
    embeddings = []
    with torch.no_grad():
        for i in range(0, len(texts), batch_size):
            batch_texts = texts[i:i + batch_size]
            inputs = tokenizer(batch_texts, return_tensors="pt", padding=True, truncation=True, max_length=512)
            inputs = {k: v.to(device) for k, v in inputs.items()}
            outputs = model(**inputs)
            batch_embeddings = outputs.last_hidden_state[:, 0, :].cpu().numpy()  # Token <s>
            embeddings.extend(batch_embeddings)
    embeddings = np.array(embeddings)
    np.save(output_file, embeddings)
    print(f"Saved doc embeddings to {output_file}")

def main():
    os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"

    tokenizer, model = load_model_and_tokenizer()
    model.to(DEVICE)
    
    datasets = [
        {"name": "Amazon_Review", "langs": ["en", "cn"]},
        {"name": "ECNews", "langs": ["en", "cn"]},
        {"name": "Rakuten_Amazon", "langs": ["en", "ja"]}
    ]
    
    for dataset in datasets:
        data_dir = os.path.join(ROOT_DIR, "data", dataset["name"])
        Path(data_dir).mkdir(parents=True, exist_ok=True)
        
        for lang in dataset["langs"]:
            # Only create embeddings for training files, not test files
            train_text_file = os.path.join(data_dir, f"train_texts_{lang}.txt")
            train_doc_output = os.path.join(data_dir, f"doc_embeddings_{lang}_train.npy")
            create_doc_embeddings(train_text_file, train_doc_output, tokenizer, model)

if __name__ == "__main__":
    main()