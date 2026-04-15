import torch
from transformers import AutoModelForSequenceClassification, Trainer, TrainingArguments
from huggingface_hub import login
import evaluate
import numpy as np

from config import *
from data import load_and_prepare_data

def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    login(token=HF_TOKEN)

    train_dataset, test_dataset, tokenizer, NUM_CLASSES, id2label, label2id = load_and_prepare_data()

    model = AutoModelForSequenceClassification.from_pretrained(
        MODEL_NAME,
        num_labels=NUM_CLASSES,
        id2label=id2label,
        label2id=label2id
    )

    for param in model.deberta.parameters():
        param.requires_grad = False

    accuracy = evaluate.load("accuracy")

    def compute_metrics(eval_pred):
        logits, labels = eval_pred
        predictions = np.argmax(logits, axis=-1)
        return accuracy.compute(predictions=predictions, references=labels)

    training_args = TrainingArguments(
        output_dir="./arxiv-classifier",
        num_train_epochs=10,
        per_device_train_batch_size=32,
        per_device_eval_batch_size=32,
        eval_strategy="epoch",
        save_strategy="epoch",
        load_best_model_at_end=True,
        warmup_steps=0.1,
        weight_decay=0.01,
        logging_steps=10,
        bf16=torch.cuda.is_bf16_supported(),
        dataloader_num_workers=0,
        dataloader_pin_memory=True,
    )

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=test_dataset,
        compute_metrics=compute_metrics,
    )
    
    trainer.train()

    model.save_pretrained(OUTPUT_PATH)
    tokenizer.save_pretrained(OUTPUT_PATH)

if __name__ == "__main__":
    main()
