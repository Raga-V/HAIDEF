import torch
import numpy as np
import os
from peft import LoraConfig, get_peft_model
from transformers import AutoTokenizer, AutoModelForCausalLM, Trainer, TrainingArguments    
from dataset_loader import train_ds, test_ds
from transformers import BitsAndBytesConfig
label2id = {"MI": 0, "Stroke": 1, "TB": 2, "Diabetes": 3}
id2label = {v: k for k, v in label2id.items()}

model_name = "google/medgemma-4b-it"

# -----------------------------
# Load Processor + Model
# -----------------------------
tokenizer = AutoTokenizer.from_pretrained(model_name)
bnb_config = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_compute_dtype=torch.bfloat16,
    bnb_4bit_use_double_quant=True,
    bnb_4bit_quant_type="nf4",
    llm_int8_enable_fp32_cpu_offload=True,  # ADD THIS
)

model = AutoModelForCausalLM.from_pretrained(
    model_name,
    quantization_config=bnb_config,
    device_map="auto"
)
model.gradient_checkpointing_enable()
model.enable_input_require_grads()

# -----------------------------
# Format Datasets
# -----------------------------
def format_for_training(example):
    text = example.get("input_text")
    label_str = example.get("label")

    if not isinstance(text, str) or not isinstance(label_str, str):
        return {
            "text": "",
            "label": -1,
            "label_str": "",
            "presentation_type": "typical"
        }

    return {
        "text": text,
        "label": label2id[label_str],
        "label_str": label_str,
        "presentation_type": example.get("presentation_type", "typical")
    }

def format_for_eval(example):
    text = example.get("input_text")
    label_str = example.get("label")

    if text is None or label_str is None:
        return {"text": None}

    label_id = label2id.get(label_str, -1)

    return {
        "text": text,
        "label": label_id,
        "label_str": label_str,
        "presentation_type": example.get("presentation_type", "typical")
    }


train_ds = train_ds.map(format_for_training)
test_ds = test_ds.map(format_for_eval)

# Now filter AFTER formatting
train_ds = train_ds.filter(lambda x: isinstance(x.get("text"), str) and len(x.get("text")) > 0)
test_ds = test_ds.filter(lambda x: isinstance(x.get("text"), str) and len(x.get("text")) > 0)

print(train_ds.column_names)
# -----------------------------
# Tokenization
# -----------------------------
def tokenize(example):
    text = example["text"]

    out = tokenizer(
        text,
        truncation=True,
        padding="max_length",
        max_length=512,
        return_token_type_ids=True,   # 🔥 REQUIRED FOR GEMMA3
    )

    out["labels"] = out["input_ids"].copy()
    return out
train_ds = train_ds.map(tokenize, remove_columns=train_ds.column_names)
test_ds = test_ds.map(tokenize, remove_columns=test_ds.column_names)

train_ds.set_format("torch")
test_ds.set_format("torch")
# -----------------------------
# Optional Smoke Test
# -----------------------------
SMOKE_TEST = os.environ.get("SMOKE_TEST", "0") in ("1", "true", "True")
if SMOKE_TEST:
    train_ds = train_ds.select(range(min(64, len(train_ds))))
    test_ds = test_ds.select(range(min(16, len(test_ds))))
    print("SMOKE_TEST active")

# -----------------------------
# Apply LoRA
# -----------------------------
lora_config = LoraConfig(
    r=8,
    lora_alpha=16,
    target_modules=["q_proj", "v_proj"],
    lora_dropout=0.05,
    bias="none",
    task_type="CAUSAL_LM"
)

model = get_peft_model(model, lora_config)
model.print_trainable_parameters()

# -----------------------------
# Training Arguments
# -----------------------------
num_epochs = 1 if SMOKE_TEST else 5

training_args = TrainingArguments(
    output_dir="./results",
    learning_rate=2e-4,
    per_device_train_batch_size=1,
    gradient_accumulation_steps=4,   # safer for 4B model
    per_device_eval_batch_size=4,
    num_train_epochs=num_epochs,
    weight_decay=0.01,
    logging_dir="./logs",
    fp16=False,
    report_to="none"
)

trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=train_ds,
    eval_dataset=test_ds,
)

trainer.train()