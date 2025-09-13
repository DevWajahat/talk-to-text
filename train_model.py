import os
import torch
from datasets import load_dataset, DatasetDict, Audio
from transformers import WhisperProcessor, WhisperForConditionalGeneration, Seq2SeqTrainingArguments, Seq2SeqTrainer, DataCollatorForSeq2Seq
from dataclasses import dataclass
from typing import Any, Dict, List, Union

@dataclass
class DataCollatorSpeechSeq2SeqWithPadding:
    processor: Any

    def __call__(self, features: List[Dict[str, Union[List[int], torch.Tensor]]]) -> Dict[str, torch.Tensor]:
        # Split inputs and labels since they have different tokenizers
        input_features = [{"input_features": feature["input_features"]} for feature in features]
        batch = self.processor.feature_extractor.pad(input_features, return_tensors="pt")

        label_features = [{"input_ids": feature["labels"]} for feature in features]
        labels_batch = self.processor.tokenizer.pad(label_features, return_tensors="pt")

        # Replace padding with -100 to ignore it in the loss calculation
        labels = labels_batch["input_ids"].masked_fill(labels_batch.attention_mask.ne(1), -100)
        
        # If bos token is appended in the training labels, it has to be removed
        if (labels[:, 0] == self.processor.tokenizer.bos_token_id).all().item():
            labels = labels[:, 1:]

        batch["labels"] = labels
        return batch

def prepare_dataset(batch):
    # load and resample audio data from 48 to 16kHz
    audio = batch["audio"]

    # compute log-Mel input features from input audio array 
    batch["input_features"] = processor.feature_extractor(
        audio["array"], sampling_rate=audio["sampling_rate"]
    ).input_features[0]

    # encode the transcriptions to label ids
    batch["labels"] = processor.tokenizer(batch["text"]).input_ids
    return batch

# --- Main Training Workflow ---
# You must run this script from the directory containing the 'datasets' folder
dataset_dir = "datasets"

try:
    data_files = {
        "train": os.path.join(dataset_dir, "train_df.csv"),
        "validation": os.path.join(dataset_dir, "validation_df.json"),
    }
    raw_dataset = load_dataset("csv", data_files=data_files)
    
except FileNotFoundError:
    print(f"Error: Dataset files not found in {dataset_dir}")
    exit()

# Load Whisper processor and model
model_name = "openai/whisper-small"
processor = WhisperProcessor.from_pretrained(model_name)
model = WhisperForConditionalGeneration.from_pretrained(model_name)

# Make sure the audio column has the correct sampling rate
raw_dataset = raw_dataset.cast_column("audio", Audio(sampling_rate=16000))

# Map the preprocessing function over the dataset
preprocessed_dataset = raw_dataset.map(
    prepare_dataset, 
    remove_columns=raw_dataset["train"].column_names, 
    num_proc=4
)

# Set up training arguments
training_args = Seq2SeqTrainingArguments(
    output_dir="./whisper-finetuned-model",
    per_device_train_batch_size=8,
    gradient_accumulation_steps=1,
    learning_rate=1e-5,
    warmup_steps=500,
    max_steps=4000,
    gradient_checkpointing=True,
    fp16=True,
    per_device_eval_batch_size=8,
    evaluation_strategy="steps",
    eval_steps=1000,
    save_steps=1000,
    push_to_hub=False,
)

# Create a data collator
data_collator = DataCollatorSpeechSeq2SeqWithPadding(processor=processor)

# Instantiate the Trainer
trainer = Seq2SeqTrainer(
    model=model,
    args=training_args,
    train_dataset=preprocessed_dataset["train"],
    eval_dataset=preprocessed_dataset["validation"],
    data_collator=data_collator,
    tokenizer=processor.tokenizer,
)

# Start training
print("Starting fine-tuning...")
trainer.train()
print("Fine-tuning complete. Saving model...")

# Save the fine-tuned model and processor
model.save_pretrained("./my-finetuned-whisper-model")
processor.save_pretrained("./my-finetuned-whisper-model")
print("Model and processor saved successfully.")