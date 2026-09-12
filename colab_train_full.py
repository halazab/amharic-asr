# Amharic ASR - Full Training Notebook
# Paste this into Google Colab with T4 GPU

# ## Cell 1: Install & Setup
!pip install -q datasets soundfile scipy huggingface_hub pyarrow

import os, gc, io, time, json, requests
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset
import numpy as np
from scipy.signal import resample
from huggingface_hub import HfApi
import pyarrow.parquet as pq
import soundfile as sf

print(f"GPU: {torch.cuda.get_device_name(0)}")
print(f"VRAM: {torch.cuda.get_device_properties(0).total_memory/1e6:.0f} MB")

# ## Cell 2: Load Dataset (5 shards = ~6700 samples)
print("Loading Amharic ASR dataset (5 shards)...")
all_data = []
for shard in range(5):
    url = f"https://huggingface.co/datasets/Harbidel/amharic-asr-merged/resolve/main/data/train-{shard:05d}-of-00050.parquet"
    r = requests.get(url, timeout=120)
    table = pq.read_table(io.BytesIO(r.content))
    df = table.to_pandas()
    for i in range(len(df)):
        row = df.iloc[i]
        audio_arr, sr = sf.read(io.BytesIO(row['audio']['bytes']))
        all_data.append({
            "audio": audio_arr.astype(np.float32),
            "sr": sr,
            "text": row['text']
        })
    print(f"  Shard {shard}: +{len(df)} (total: {len(all_data)})")

print(f"Loaded {len(all_data)} training samples")

# ## Cell 3: Tokenizer
class Tok:
    def __init__(self):
        self.c2i = {"<b>": 0, "<u>": 1}
        chars = list("ሀለሐመሠረሰሸቀበተቸኀነአከወዘየደደጀገገጠጰጸፀፈፐאבగდვზთიკლმნოპჟრსტუფქღყშჩცძწჭხჯჰ0123456789.,!? ")
        for i, c in enumerate(chars[:1022]): self.c2i[c] = i+2
        self.i2c = {v:k for k,v in self.c2i.items()}
    def enc(self, t): return [self.c2i.get(c,1) for c in t]
    def decode(self, ids): return "".join([self.i2c.get(i,"") for i in ids if i>0])

tok = Tok()
print(f"Vocab: {len(tok.c2i)} tokens")

# ## Cell 4: Dataset
MAX_AUDIO = 24000  # 1.5s max

class ASRDataset(Dataset):
    def __init__(self, data):
        self.data = data
    def __len__(self):
        return len(self.data)
    def __getitem__(self, i):
        s = self.data[i]
        audio = s["audio"]
        sr = s["sr"]
        if sr != 16000:
            audio = resample(audio, int(len(audio) / sr * 16000))
        audio = audio[:MAX_AUDIO]
        mx = np.max(np.abs(audio))
        if mx > 0: audio /= mx
        if len(audio) < 1600:
            audio = np.pad(audio, (0, 1600 - len(audio)))
        return {"audio": torch.FloatTensor(audio), "text": s["text"]}

def collate_fn(batch):
    audios = [b["audio"] for b in batch]
    max_len = max(len(a) for a in audios)
    padded = torch.zeros(len(audios), max_len)
    for i, a in enumerate(audios):
        padded[i, :len(a)] = a
    return {"audio": padded, "text": [b["text"] for b in batch]}

# ## Cell 5: Model (5.8M params - fits T4)
class Prep(nn.Module):
    def __init__(self):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv1d(1,64,64,stride=2,padding=32), nn.BatchNorm1d(64), nn.ReLU(),
            nn.Conv1d(64,128,32,stride=3,padding=16), nn.BatchNorm1d(128), nn.ReLU(),
            nn.Conv1d(128,256,16,stride=2,padding=8), nn.BatchNorm1d(256), nn.ReLU())
    def forward(self, x): return self.net(x.unsqueeze(1)).transpose(1,2)

class ASR(nn.Module):
    def __init__(self):
        super().__init__()
        self.prep = Prep()
        self.enc = nn.TransformerEncoder(
            nn.TransformerEncoderLayer(256,4,1024,0.1,batch_first=True), 6)
        self.head = nn.Linear(256, 1024)
    def forward(self, x): return self.head(self.enc(self.prep(x)))

device = torch.device("cuda")
model = ASR().to(device)
print(f"Model: {sum(p.numel() for p in model.parameters()):,} params")

# ## Cell 6: Training (30 epochs, early stopping)
train_ds = ASRDataset(all_data)
train_dl = DataLoader(train_ds, batch_size=8, shuffle=True,
                      collate_fn=collate_fn, pin_memory=True)

optimizer = optim.AdamW(model.parameters(), lr=2e-3, weight_decay=0.01)
scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=30)
criterion = nn.CTCLoss(blank=0, zero_infinity=True)

print(f"Training: {len(all_data)} samples, batch=8, {len(train_dl)} steps/epoch")
os.makedirs("checkpoints", exist_ok=True)
best_loss = float("inf")
patience = 0

for epoch in range(30):
    t0 = time.time()
    model.train()
    total_loss = 0
    n = 0
    for i, batch in enumerate(train_dl):
        audio = batch["audio"].to(device)
        texts = batch["text"]
        logits = model(audio)
        targets = [tok.enc(t) for t in texts]
        t_lens = [len(t) for t in targets]
        max_t = max(t_lens) if t_lens else 1
        padded = torch.zeros(len(targets), max_t, dtype=torch.long).to(device)
        for j, t in enumerate(targets):
            padded[j, :len(t)] = torch.tensor(t)
        log_probs = torch.log_softmax(logits, dim=-1)
        loss = criterion(log_probs.permute(1,0,2), padded,
                        torch.full((audio.size(0),), logits.size(1), dtype=torch.long, device=device),
                        torch.tensor(t_lens, device=device))
        optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 5.0)
        optimizer.step()
        total_loss += loss.item()
        n += 1
        if i % 50 == 0:
            print(f"  [{epoch}] {i}/{len(train_dl)} loss={loss.item():.3f}")
    scheduler.step()
    avg = total_loss / max(n, 1)
    elapsed = time.time() - t0
    print(f"Epoch {epoch+1}/30: loss={avg:.4f} time={elapsed:.0f}s")
    if avg < best_loss:
        best_loss = avg
        patience = 0
        torch.save({
            "epoch": epoch, "model_state_dict": model.state_dict(),
            "loss": avg, "vocab_size": len(tok.c2i),
        }, "checkpoints/best_model.pt")
        print(f"  BEST -> saved")
    else:
        patience += 1
        if patience >= 5:
            print("Early stopping!")
            break

torch.save(model.state_dict(), "checkpoints/final_model.pt")
print(f"\nDone! Best loss: {best_loss:.4f}")

# ## Cell 7: Push to HuggingFace
hf_token = os.environ.get("HF_TOKEN", "YOUR_HF_TOKEN_HERE")
api = HfApi()
repo_id = "halazab21/amharic-asr-v1"

api.create_repo(repo_id=repo_id, exist_ok=True)
api.upload_file(
    path_or_fileobj="checkpoints/best_model.pt",
    path_in_repo="best_model.pt",
    repo_id=repo_id, repo_type="model")
print(f"Uploaded: https://huggingface.co/{repo_id}")

# Also push config
with open("config.json", "w") as f:
    json.dump({"vocab_size": len(tok.c2i), "model": "ASR-5.8M", "max_audio": MAX_AUDIO,
               "sample_rate": 16000, "train_samples": len(all_data), "best_loss": best_loss}, f)
api.upload_file(path_or_fileobj="config.json", path_in_repo="config.json",
                repo_id=repo_id, repo_type="model")
print("Config uploaded!")

# ## Cell 8: Test Inference
model.eval()
test_audio = torch.randn(1, 16000).to(device)  # 1 second random
with torch.no_grad():
    logits = model(test_audio)
    pred = torch.argmax(logits, dim=-1)
    text = tok.decode(pred[0].cpu().numpy())
    print(f"Test output: '{text}'")
print("Model working!")
