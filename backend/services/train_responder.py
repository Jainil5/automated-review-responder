import torch
from datasets import load_dataset
from transformers import (
    AutoTokenizer,
    AutoModelForSeq2SeqLM,
    TrainingArguments,
    Trainer,
    DataCollatorForSeq2Seq
)
from peft import LoraConfig, get_peft_model, TaskType

if torch.backends.mps.is_available():
    device = torch.device("mps")
    print("Using MPS")
else:
    device = torch.device("cpu")
    print("Using CPU")

dataset = load_dataset(
    "csv",
    data_files="backend/datasets/reviews-data.csv"
)

dataset = dataset["train"].train_test_split(
    test_size=0.2,
    seed=42
)

MODEL_NAME = "google/flan-t5-base"

tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)

model = AutoModelForSeq2SeqLM.from_pretrained(MODEL_NAME)
model.to(device)

lora_config = LoraConfig(
    task_type=TaskType.SEQ_2_SEQ_LM,
    r=8,
    lora_alpha=16,
    lora_dropout=0.1,
    target_modules=["q", "v"]  
)

model = get_peft_model(model, lora_config)
model.print_trainable_parameters()

MAX_INPUT_LENGTH = 256
MAX_TARGET_LENGTH = 128

def preprocess(batch):
    prompts = [
        f"Respond professionally to this customer review on behalf of company.:\n{review}"
        for review in batch["review"]
    ]

    inputs = tokenizer(
        prompts,
        truncation=True,
        padding="max_length",
        max_length=MAX_INPUT_LENGTH
    )

    targets = tokenizer(
        batch["response"],
        truncation=True,
        padding="max_length",
        max_length=MAX_TARGET_LENGTH
    )

    inputs["labels"] = targets["input_ids"]
    return inputs

tokenized_ds = dataset.map(
    preprocess,
    batched=True,
    remove_columns=dataset["train"].column_names
)

data_collator = DataCollatorForSeq2Seq(
    tokenizer=tokenizer,
    model=model
)

training_args = TrainingArguments(
    output_dir="backend/models/flan_t5_review_lora",
    per_device_train_batch_size=2,
    gradient_accumulation_steps=4,
    learning_rate=2e-4,
    num_train_epochs=10,
    logging_steps=50,
    fp16=False,
    bf16=False,
    dataloader_num_workers=0,
    report_to="none"
)

trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=tokenized_ds["train"],
    eval_dataset=tokenized_ds["test"],
    data_collator=data_collator
)

trainer.train()

model.save_pretrained("backend/models/flan_t5_review_lora")
tokenizer.save_pretrained("backend/models/flan_t5_review_lora")

print("Fine-tuning completed successfully!")

# def generate_response(review_text):
#     prompt = f"Respond politely and professionally to this customer review:\n{review_text}"
#     inputs = tokenizer(prompt, return_tensors="pt").to(device)

#     outputs = model.generate(
#         **inputs,
#         max_new_tokens=100,
#         temperature=0.7
#     )

#     return tokenizer.decode(outputs[0], skip_special_tokens=True)

# test_review = "The zipper on my jacket keeps getting stuck."
# print("\nREVIEW:", test_review)
# print("RESPONSE:", generate_response(test_review))
# print("-"*20)
