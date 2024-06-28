import os
import json
import numpy as np
import torch, platform, wandb, random
from transformers import AutoModelForCausalLM, AutoTokenizer, Trainer, TrainingArguments, TrainerCallback, BitsAndBytesConfig
from datasets import Dataset, DatasetDict
from peft import LoraConfig
from trl import SFTTrainer

api_key = "4556651ce2176ed46d63906314d5c7703d3ff741"
wandb.login(key=api_key)
wandb.init(project="llama-reasoning")

# Function to load and preprocess data
def json_to_string(input_files, proof_files, ontology_path):
    premises, proofs = [], []
    for input_f, proof_f in zip(input_files, proof_files):
        input_file_path = os.path.join(ontology_path, input_f)
        proof_file_path = os.path.join(ontology_path, proof_f)
        
        try:
            with open(input_file_path, 'r') as f:
                premise_string = json.dumps(json.load(f))
            premises.append(premise_string)
        except json.JSONDecodeError as e:
            print(f"Error loading JSON from {input_file_path}: {str(e)}")

        try:
            with open(proof_file_path, 'r') as f:
                proof_string = json.dumps(json.load(f))
            proofs.append(proof_string)
        except json.JSONDecodeError as e:
            print(f"Error loading JSON from {proof_file_path}: {str(e)}")

    return premises, proofs

def load_ontology(path, train_ratio=0.7, val_ratio=0.1):
    premises, proofs = [], []
    
    print(f"Path: {path}")
    for dir in os.listdir(path):
        ontology_path = os.path.join(path, dir)
        input_files = sorted([f for f in os.listdir(ontology_path) if f.startswith("Input")])
        proof_files = sorted([f for f in os.listdir(ontology_path) if f.startswith('Proof')])
        cur_dir_premises, cur_dir_proofs = json_to_string(input_files, proof_files, ontology_path)
        
        premises.extend(cur_dir_premises)
        proofs.extend(cur_dir_proofs)
        
        print(f"Current dir: {dir}: {len(cur_dir_premises)}")
        assert len(premises) == len(proofs)

    data_length = len(proofs)
    train_len = int(data_length * train_ratio)
    val_len = int(train_len * val_ratio)

    indices = np.arange(data_length)
    np.random.shuffle(indices)

    train_indices = indices[:train_len]
    val_indices = indices[train_len:train_len + val_len]
    test_indices = indices[train_len + val_len:]

    train_data = [{"premise": premises[i], "proof": proofs[i]} for i in train_indices]
    val_data = [{"premise": premises[i], "proof": proofs[i]} for i in val_indices]
    test_data = [{"premise": premises[i], "proof": proofs[i]} for i in test_indices]

    return train_data, val_data, test_data

# Load and prepare data
data_path = "test_dataset"
train_data, val_data, test_data = load_ontology(data_path)

# Convert to Dataset objects
train_dataset = Dataset.from_list(train_data)
val_dataset = Dataset.from_list(val_data)
test_dataset = Dataset.from_list(test_data)

datasets = DatasetDict({
    'train': train_dataset,
    'validation': val_dataset,
    'test': test_dataset
})

# 4-bit Quantization Configuration
compute_dtype = getattr(torch, "float16")
quant_config = BitsAndBytesConfig(load_in_4bit=True, bnb_4bit_quant_type="nf4", bnb_4bit_compute_dtype=compute_dtype, bnb_4bit_use_double_quant=False)

# Pre trained model
model_name = "meta-llama/Llama-2-7b-hf"

# Hugging face repository link to save fine-tuned model(Create new repository in huggingface,copy and paste here)
access_token = "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"
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Load the model and tokenizer
model = AutoModelForCausalLM.from_pretrained(
    model_name,
    use_auth_token=access_token,
    device_map='auto' if device.type == 'cuda' else None
)
model.config.use_cache = False
model.config.pretraining_tp = 1


tokenizer = AutoTokenizer.from_pretrained(model_name, token=access_token)
tokenizer.pad_token = tokenizer.eos_token
tokenizer.padding_side = "right"

# Tokenize data
def tokenize_function(example):
    return tokenizer(example["premise"], example["proof"], truncation=True, max_length=1024, padding="max_length")

tokenized_datasets = datasets.map(tokenize_function, batched=True)

# Set the format for PyTorch
tokenized_datasets.set_format(type="torch", columns=["input_ids", "attention_mask"])

peft_params = LoraConfig(lora_alpha=16, lora_dropout=0.1, r=64, bias="none", task_type="CAUSAL_LM")

# Create the text generation callback
class TextGenerationCallback(TrainerCallback):
    def __init__(self, tokenizer, model, val_dataset, max_length=512, interval=10):
        self.tokenizer = tokenizer
        self.model = model
        self.val_dataset = val_dataset
        self.max_length = max_length
        self.interval = interval

    def on_step_end(self, args, state, control, **kwargs):
        if state.global_step % self.interval == 0 and state.global_step > 0:
            self.model.eval()
            for _ in range(2):  # Generate text for 2 examples
                example = random.choice(self.val_dataset)
                inputs = self.tokenizer(example["premise"], return_tensors="pt")
                inputs = {k: v.to(args.device) for k, v in inputs.items()}
                with torch.no_grad():
                    outputs = self.model.generate(**inputs, max_length=self.max_length)
                generated_text = self.tokenizer.decode(outputs[0], skip_special_tokens=True)
                print(f"Step {state.global_step}")
                print(f"Input: {example['premise']}")
                print(f"Generated: {generated_text}")
            self.model.train()

# Create the text generation callback
text_gen_callback = TextGenerationCallback(tokenizer, model, tokenized_datasets["validation"])

# Training arguments
training_args = TrainingArguments(
    output_dir="./results",
    evaluation_strategy="epoch",
    learning_rate=2e-5,
    per_device_train_batch_size=1,
    per_device_eval_batch_size=1,
    num_train_epochs=3,
    weight_decay=0.01,
    logging_dir='./logs',
    logging_steps=10,
    report_to="wandb",  # Enable WandB logging
    run_name="llama-reasoning"  # Name of the WandB run
)

# SFTTrainer
trainer = SFTTrainer(
    model=model,
    args=training_args,
    train_dataset=tokenized_datasets["train"],
    eval_dataset=tokenized_datasets["validation"],
    callbacks=[text_gen_callback]  # Add the callback to the trainer
)

# Train the model
trainer.train()

# Evaluate the model
trainer.evaluate()
wandb.finish()

