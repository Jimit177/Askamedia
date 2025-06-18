
from transformers import T5Tokenizer, T5ForConditionalGeneration, Trainer, TrainingArguments, DataCollatorForSeq2Seq
from datasets import load_dataset

# Load dataset
dataset = load_dataset("json", data_files="../data/flan_finetune_local_qa_200.jsonl", split="train")

# Model and tokenizer
model_name = "google/flan-t5-small"
tokenizer = T5Tokenizer.from_pretrained(model_name)
model = T5ForConditionalGeneration.from_pretrained(model_name)

# Tokenization function
def preprocess(example):
    model_inputs = tokenizer(example["input"], max_length=256, truncation=True, padding="max_length")
    with tokenizer.as_target_tokenizer():
        labels = tokenizer(example["output"], max_length=128, truncation=True, padding="max_length")
    model_inputs["labels"] = labels["input_ids"]
    return model_inputs

# Tokenize dataset
tokenized_dataset = dataset.map(preprocess, batched=True)

# Training arguments
training_args = TrainingArguments(
    output_dir="../Model/flan_t5_finetuned_model",
    per_device_train_batch_size=4,
    num_train_epochs=5,
    learning_rate=5e-5,
    weight_decay=0.01,
    logging_dir="./logs",
    logging_steps=10,
    save_strategy="epoch",
    save_total_limit=1,
    report_to="none"
)



# Data collator
data_collator = DataCollatorForSeq2Seq(tokenizer, model=model)

# Trainer
trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=tokenized_dataset,
    tokenizer=tokenizer,
    data_collator=data_collator
)

# Start training
trainer.train()

# Save the model
output_dir="../Model/flan_t5_finetuned_model"
...
model.save_pretrained("../Model/flan_t5_finetuned_model")
tokenizer.save_pretrained("../Model/flan_t5_finetuned_model")

