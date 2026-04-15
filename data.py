import kagglehub
from kagglehub import KaggleDatasetAdapter
import pandas as pd
from sklearn.model_selection import train_test_split
from transformers import AutoTokenizer
import torch
from torch.utils.data import Dataset

from config import *

class ArxivDataset(Dataset):
    def __init__(self, texts, labels, tokenizer, length):
        self.encodings = tokenizer(
            texts.tolist(),
            truncation=True,
            padding="max_length",
            max_length=length,
        )
        self.labels = labels.tolist()

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        return {
            "input_ids": torch.tensor(self.encodings["input_ids"][idx]),
            "attention_mask": torch.tensor(self.encodings["attention_mask"][idx]),
            "labels": torch.tensor(self.labels[idx])
        }

def load_and_prepare_data():
    df = kagglehub.dataset_load(
        KaggleDatasetAdapter.PANDAS,
        DATASET_NAME,
        "arxiv-metadata-oai-snapshot.json",
        pandas_kwargs={"lines": True, "nrows": NUM_ROWS}
    )

    df = df.dropna(subset=["title", "abstract", "categories"])
    df["label_name"] = df["categories"].str.split().str[0].str.split(".").str[0]
    df["text"] = df["title"] + " [SEP] " + df["abstract"]

    unique_labels = sorted(df["label_name"].unique())
    NUM_CLASSES = len(unique_labels)
    label2id = {label: idx for idx, label in enumerate(unique_labels)}
    id2label = {idx: label for label, idx in label2id.items()}
    df["label"] = df["label_name"].map(label2id)

    train_df, test_df = train_test_split(
        df,
        test_size=TEST_SIZE,
        random_state=RANDOM_STATE,
        stratify=df["label"]
    )

    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)

    train_dataset = ArxivDataset(train_df["text"], train_df["label"], tokenizer, MAX_LENGTH)
    test_dataset = ArxivDataset(test_df["text"], test_df["label"], tokenizer, MAX_LENGTH)


    return train_dataset, test_dataset, tokenizer, NUM_CLASSES, id2label, label2id