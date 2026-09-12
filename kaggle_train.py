#!/usr/bin/env python3
"""
Kaggle Training Notebook for Amharic ASR
Run this on Kaggle with GPU accelerator (T4)
"""

# Cell 1: Install dependencies
!pip install -q datasets soundfile
!pip install -q torch torchaudio
!pip install -q sentencepiece
!pip install -q huggingface_hub

# Cell 2: Import libraries
import os
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset
import numpy as np
import soundfile as sf
from datasets import load_dataset
from pathlib import Path
import time
import json

print(f"PyTorch version: {torch.__version__}")
print(f"CUDA available: {torch.cuda.is_available()}")
print(f"GPU: {torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'None'}")

# Cell 3: Configuration
CONFIG = {
    "model_name": "amharic_asr_v1",
    "vocab_size": 1024,
    "sample_rate": 16000,
    
    # Model architecture
    "encoder_dim": 256,
    "encoder_layers": 6,
    "attention_heads": 4,
    "ffn_dim": 1024,
    
    # Training
    "batch_size": 16,
    "gradient_accumulation": 2,
    "learning_rate": 2e-3,
    "warmup_steps": 2000,
    "max_epochs": 50,
    "early_stopping_patience": 5,
    
    # Progressive training
    "patch_size_hours": 20.0,
    "total_patches": 10,
    
    # Augmentation
    "speed_perturbation": [0.9, 0.95, 1.0, 1.05, 1.1],
    "noise_augmentation": True,
    
    # Checkpointing
    "save_every_steps": 500,
    "keep_last_n_checkpoints": 5,
    
    # HuggingFace
    "dataset_name": "Harbidel/amharic-asr-merged",
    
    # Output
    "output_dir": "./checkpoints",
    "model_dir": "./models"
}

os.makedirs(CONFIG["output_dir"], exist_ok=True)
os.makedirs(CONFIG["model_dir"], exist_ok=True)

print("Config:", json.dumps(CONFIG, indent=2))

# Cell 4: Load dataset from HuggingFace
print("Loading dataset from HuggingFace...")
dataset = load_dataset(CONFIG["dataset_name"])

print(f"Dataset loaded!")
print(f"Train: {len(dataset['train'])} samples")
print(f"Validation: {len(dataset['validation'])} samples")
print(f"Test: {len(dataset['test'])} samples")

# Sample to check structure
sample = dataset["train"][0]
print(f"\nSample structure:")
print(f"  Audio: {sample['audio']}")
print(f"  Text: {sample['text']}")
print(f"  Source: {sample['source']}")

# Cell 5: Audio Dataset class
class AmharicASRDataset(Dataset):
    def __init__(self, dataset, config, split="train"):
        self.dataset = dataset[split]
        self.config = config
        self.sample_rate = config["sample_rate"]
        
    def __len__(self):
        return len(self.dataset)
    
    def __getitem__(self, idx):
        sample = self.dataset[idx]
        
        # Get audio
        audio = sample["audio"]
        audio_array = np.array(audio["array"], dtype=np.float32)
        orig_sr = audio["sampling_rate"]
        
        # Resample to target SR
        if orig_sr != self.sample_rate:
            from scipy.signal import resample
            duration = len(audio_array) / orig_sr
            target_length = int(duration * self.sample_rate)
            audio_array = resample(audio_array, target_length)
        
        # Normalize
        if np.max(np.abs(audio_array)) > 0:
            audio_array = audio_array / np.max(np.abs(audio_array))
        
        # Get transcript
        text = sample["text"]
        
        return {
            "audio": torch.FloatTensor(audio_array),
            "text": text,
            "audio_length": len(audio_array),
            "source": sample["source"]
        }

# Cell 6: Tokenizer (simple character-level for now)
class SimpleTokenizer:
    def __init__(self, vocab_size=1024):
        self.vocab_size = vocab_size
        self.char_to_idx = {}
        self.idx_to_char = {}
        
        # Add special tokens
        self.char_to_idx["<blank>"] = 0
        self.char_to_idx["<unk>"] = 1
        self.char_to_idx["<space>"] = 2
        
        # Add Ge'ez characters
        ge_ez_chars = list("ሀለሐመሠረሰሸቀበተቸኀነአከወዘየደደጀገገጠጰጸፀፈፐאבગდვზთიკლმნოპჟრსტუფქღყშჩცძწჭხჯჰ")
        
        # Add numbers and punctuation
        extra_chars = list("0123456789.,!?;:'\"() ")
        
        # Build vocabulary
        idx = 3
        for char in ge_ez_chars + extra_chars:
            if idx < vocab_size:
                self.char_to_idx[char] = idx
                self.idx_to_char[idx] = char
                idx += 1
        
        self.vocab_size = min(vocab_size, idx)
    
    def encode(self, text):
        return [self.char_to_idx.get(c, 1) for c in text]  # 1 = <unk>
    
    def decode(self, indices):
        return "".join([self.idx_to_char.get(i, "") for i in indices if i > 0])

# Cell 7: Model Architecture (Moonshine-style)
class AudioPreprocessor(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = nn.Conv1d(1, 64, kernel_size=64, stride=2, padding=32)
        self.conv2 = nn.Conv1d(64, 128, kernel_size=32, stride=3, padding=16)
        self.conv3 = nn.Conv1d(128, 256, kernel_size=16, stride=2, padding=8)
        self.bn1 = nn.BatchNorm1d(64)
        self.bn2 = nn.BatchNorm1d(128)
        self.bn3 = nn.BatchNorm1d(256)
        self.relu = nn.ReLU()
        
    def forward(self, x):
        x = x.unsqueeze(1)  # (B, 1, T)
        x = self.relu(self.bn1(self.conv1(x)))
        x = self.relu(self.bn2(self.conv2(x)))
        x = self.relu(self.bn3(self.conv3(x)))
        return x.transpose(1, 2)  # (B, T', 256)

class TransformerEncoder(nn.Module):
    def __init__(self, dim=256, layers=6, heads=4, ffn_dim=1024, dropout=0.1):
        super().__init__()
        self.layers = nn.ModuleList([
            nn.TransformerEncoderLayer(
                d_model=dim,
                nhead=heads,
                dim_feedforward=ffn_dim,
                dropout=dropout,
                batch_first=True
            )
            for _ in range(layers)
        ])
        self.norm = nn.LayerNorm(dim)
        
    def forward(self, x):
        for layer in self.layers:
            x = layer(x)
        return self.norm(x)

class AmharicASR(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.config = config
        
        # Audio preprocessor
        self.preprocessor = AudioPreprocessor()
        
        # Transformer encoder
        self.encoder = TransformerEncoder(
            dim=config["encoder_dim"],
            layers=config["encoder_layers"],
            heads=config["attention_heads"],
            ffn_dim=config["ffn_dim"]
        )
        
        # CTC head
        self.ctc_head = nn.Linear(config["encoder_dim"], config["vocab_size"])
        
    def forward(self, audio):
        # Preprocess audio
        features = self.preprocessor(audio)
        
        # Encode
        encoded = self.encoder(features)
        
        # CTC projection
        logits = self.ctc_head(encoded)
        
        return logits

# Cell 8: CTC Loss and decoding
def ctc_decode(logits, tokenizer, blank=0):
    """CTC greedy decoding."""
    indices = torch.argmax(logits, dim=-1)
    
    # Remove blanks and duplicates
    decoded = []
    prev_idx = -1
    for idx in indices:
        if idx != blank and idx != prev_idx:
            decoded.append(idx.item())
        prev_idx = idx
    
    return tokenizer.decode(decoded)

# Cell 9: Training loop
def train_epoch(model, dataloader, optimizer, criterion, device, config):
    model.train()
    total_loss = 0
    num_batches = 0
    
    for batch_idx, batch in enumerate(dataloader):
        audio = batch["audio"].to(device)
        texts = batch["text"]
        
        # Forward pass
        logits = model(audio)
        
        # Prepare CTC targets
        tokenizer = SimpleTokenizer(config["vocab_size"])
        targets = [tokenizer.encode(t) for t in texts]
        target_lengths = [len(t) for t in targets]
        
        # Pad targets
        max_target_len = max(target_lengths)
        padded_targets = torch.zeros(len(targets), max_target_len, dtype=torch.long)
        for i, t in enumerate(targets):
            padded_targets[i, :len(t)] = torch.tensor(t)
        
        # CTC loss
        input_lengths = torch.full((audio.size(0),), logits.size(1), dtype=torch.long)
        target_lengths = torch.tensor(target_lengths, dtype=torch.long)
        
        log_probs = torch.log_softmax(logits, dim=-1)
        loss = criterion(log_probs.permute(1, 0, 2), padded_targets, input_lengths, target_lengths)
        
        # Backward pass
        optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 5.0)
        optimizer.step()
        
        total_loss += loss.item()
        num_batches += 1
        
        if batch_idx % 100 == 0:
            print(f"  Batch {batch_idx}/{len(dataloader)}, Loss: {loss.item():.4f}")
    
    return total_loss / num_batches

# Cell 10: Main training
def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    
    # Load dataset
    print("Loading dataset...")
    dataset = load_dataset(CONFIG["dataset_name"])
    
    # Create data loaders
    train_dataset = AmharicASRDataset(dataset, CONFIG, "train")
    val_dataset = AmharicASRDataset(dataset, CONFIG, "validation")
    
    train_loader = DataLoader(train_dataset, batch_size=CONFIG["batch_size"], shuffle=True, num_workers=2)
    val_loader = DataLoader(val_dataset, batch_size=CONFIG["batch_size"], shuffle=False, num_workers=2)
    
    # Initialize model
    model = AmharicASR(CONFIG).to(device)
    print(f"Model parameters: {sum(p.numel() for p in model.parameters()):,}")
    
    # Initialize tokenizer
    tokenizer = SimpleTokenizer(CONFIG["vocab_size"])
    
    # Initialize optimizer and loss
    optimizer = optim.AdamW(model.parameters(), lr=CONFIG["learning_rate"], weight_decay=0.01)
    criterion = nn.CTCLoss(blank=0, zero_infinity=True)
    
    # Learning rate scheduler
    scheduler = optim.lr_scheduler.OneCycleLR(
        optimizer,
        max_lr=CONFIG["learning_rate"],
        epochs=CONFIG["max_epochs"],
        steps_per_epoch=len(train_loader)
    )
    
    # Training loop
    best_val_loss = float("inf")
    patience_counter = 0
    
    print("\nStarting training...")
    for epoch in range(CONFIG["max_epochs"]):
        print(f"\nEpoch {epoch+1}/{CONFIG['max_epochs']}")
        
        # Train
        train_loss = train_epoch(model, train_loader, optimizer, criterion, device, CONFIG)
        print(f"Train Loss: {train_loss:.4f}")
        
        # Save checkpoint
        if (epoch + 1) % 5 == 0:
            checkpoint_path = f"{CONFIG['output_dir']}/checkpoint_epoch_{epoch+1}.pt"
            torch.save({
                "epoch": epoch,
                "model_state_dict": model.state_dict(),
                "optimizer_state_dict": optimizer.state_dict(),
                "train_loss": train_loss,
                "config": CONFIG
            }, checkpoint_path)
            print(f"Saved checkpoint: {checkpoint_path}")
        
        # Early stopping check
        if train_loss < best_val_loss:
            best_val_loss = train_loss
            patience_counter = 0
            
            # Save best model
            best_model_path = f"{CONFIG['output_dir']}/best_model.pt"
            torch.save(model.state_dict(), best_model_path)
            print(f"Saved best model: {best_model_path}")
        else:
            patience_counter += 1
            if patience_counter >= CONFIG["early_stopping_patience"]:
                print("Early stopping triggered!")
                break
    
    print("\nTraining complete!")
    
    # Save final model
    final_model_path = f"{CONFIG['model_dir']}/amharic_asr_final.pt"
    torch.save(model.state_dict(), final_model_path)
    print(f"Saved final model: {final_model_path}")

# Cell 11: Run training
if __name__ == "__main__":
    main()
