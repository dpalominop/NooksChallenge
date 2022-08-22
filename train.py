import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

from datasets import Dataset
from transformers import AutoTokenizer
from transformers import DataCollatorWithPadding
from transformers import AutoModelForSequenceClassification, TrainingArguments, Trainer
from utils.split import train_validate_test_split
from utils.metrics import compute_metrics

train_df = pd.read_csv('data/train_data.csv')

train_df["label"]=train_df["label"].map({"human":1, "voicemail":0})
train_df = train_df.dropna(subset=["transcript"])

train_df, eval_df = train_validate_test_split(train_df)

train_ds = Dataset.from_pandas(train_df[["transcript", "label"]])
eval_ds = Dataset.from_pandas(eval_df[["transcript", "label"]])

model_name = "bert-base-uncased"
tokenizer = AutoTokenizer.from_pretrained(model_name)
def preprocess_function(examples):
    return tokenizer(examples["transcript"], truncation=True)

tokenized_train = train_ds.map(preprocess_function, batched=True)
tokenized_eval = eval_ds.map(preprocess_function, batched=True)

data_collator = DataCollatorWithPadding(tokenizer=tokenizer)
model = AutoModelForSequenceClassification.from_pretrained(model_name, num_labels=2)

training_args = TrainingArguments(
    output_dir="./results",
    evaluation_strategy = "epoch",
    save_strategy = "epoch",
    learning_rate=2e-5,
    per_device_train_batch_size=64,
    per_device_eval_batch_size=64,
    num_train_epochs=5,
    weight_decay=0.01,
    load_best_model_at_end=True,
    metric_for_best_model="f1",
    run_name="nooks",
)

trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=tokenized_train,
    eval_dataset=tokenized_eval,
    tokenizer=tokenizer,
    data_collator=data_collator,
    compute_metrics=compute_metrics
)

trainer.train()
trainer.save_model("results/best_model")
