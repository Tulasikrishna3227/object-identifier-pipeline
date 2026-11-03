"""
Transfer learning: fine-tune a pretrained EfficientNet-B0 to identify 37 pet breeds.
Run in Colab with GPU (Runtime > Change runtime type > T4 GPU).

!pip install datasets
Steps: data -> pretrained model -> phase 1 (train head only) -> phase 2 (fine-tune all) -> evaluate -> predict
Expected: ~90-94% test accuracy in about 10 minutes.
"""
import torch, torch.nn as nn, torch.optim as optim
import torchvision, torchvision.transforms as T
from torch.utils.data import DataLoader, Subset
import matplotlib.pyplot as plt
import numpy as np
from datasets import load_dataset
from sklearn.metrics import confusion_matrix
from PIL import Image

torch.manual_seed(42)
device = "cuda" if torch.cuda.is_available() else "cpu"
print("Using:", device)

# ---------- 1. Data ----------
hf = load_dataset("timm/oxford-iiit-pet")            # 37 breeds, real full-size photos
CLASSES = hf["train"].features["label"].names
print("Classes:", len(CLASSES), "| train:", len(hf["train"]), "| test:", len(hf["test"]))

# Pretrained models expect ImageNet normalization and 224x224 input
MEAN, STD = (0.485, 0.456, 0.406), (0.229, 0.224, 0.225)
train_tf = T.Compose([
    T.RandomResizedCrop(224, scale=(0.7, 1.0)),
    T.RandomHorizontalFlip(),
    T.ColorJitter(0.2, 0.2, 0.2),
    T.ToTensor(), T.Normalize(MEAN, STD),
])
eval_tf = T.Compose([T.Resize(256), T.CenterCrop(224), T.ToTensor(), T.Normalize(MEAN, STD)])

class HFImages(torch.utils.data.Dataset):
    def __init__(self, split, tf):
        self.d, self.tf = split, tf
    def __len__(self):
        return len(self.d)
    def __getitem__(self, i):
        r = self.d[i]
        return self.tf(r["image"].convert("RGB")), r["label"]

# 90/10 train/val split (test set stays untouched until the end)
perm = np.random.RandomState(42).permutation(len(hf["train"]))
n_val = int(0.1 * len(perm))
val_idx, train_idx = perm[:n_val].tolist(), perm[n_val:].tolist()
train_set = Subset(HFImages(hf["train"], train_tf), train_idx)
val_set = Subset(HFImages(hf["train"], eval_tf), val_idx)
test_set = HFImages(hf["test"], eval_tf)

train_loader = DataLoader(train_set, batch_size=32, shuffle=True, num_workers=2)
val_loader = DataLoader(val_set, batch_size=64, num_workers=2)
test_loader = DataLoader(test_set, batch_size=64, num_workers=2)

# ---------- 2. Pretrained model ----------
model = torchvision.models.efficientnet_b0(weights="DEFAULT")   # already knows edges, textures, shapes from ImageNet
in_features = model.classifier[1].in_features
model.classifier[1] = nn.Linear(in_features, len(CLASSES))     # replace the last layer: 1000 classes -> 37 breeds
model = model.to(device)

criterion = nn.CrossEntropyLoss(label_smoothing=0.1)            # label smoothing = less overconfidence
scaler = torch.amp.GradScaler("cuda", enabled=(device == "cuda"))  # mixed precision = faster on T4

def run_epoch(loader, optimizer=None, scheduler=None):
    train = optimizer is not None
    model.train(train)
    total_loss, correct, n = 0, 0, 0
    with torch.set_grad_enabled(train):
        for x, y in loader:
            x, y = x.to(device), y.to(device)
            with torch.autocast(device_type="cuda", dtype=torch.float16, enabled=(device == "cuda")):
                out = model(x)
                loss = criterion(out, y)
            if train:
                optimizer.zero_grad()
                scaler.scale(loss).backward()
                scaler.step(optimizer)
                scaler.update()
                if scheduler: scheduler.step()
            total_loss += loss.item() * len(y)
            correct += (out.argmax(1) == y).sum().item()
            n += len(y)
    return total_loss / n, correct / n

history = {"tr_loss": [], "va_loss": [], "tr_acc": [], "va_acc": []}
best_val, best_state = 0, None

def fit(epochs, optimizer, scheduler, tag):
    global best_val, best_state
    for ep in range(epochs):
        tl, ta = run_epoch(train_loader, optimizer, scheduler)
        vl, va = run_epoch(val_loader)
        for k, v in zip(history, (tl, vl, ta, va)):
            history[k].append(v)
        if va > best_val:
            best_val, best_state = va, {k: v.clone() for k, v in model.state_dict().items()}
        print(f"[{tag}] epoch {ep+1}/{epochs}  train loss {tl:.3f} acc {ta:.3f} | val loss {vl:.3f} acc {va:.3f}")

# ---------- 3. Phase 1: train only the new head (backbone frozen) ----------
for p in model.features.parameters():
    p.requires_grad = False
opt = optim.AdamW(model.classifier.parameters(), lr=1e-3, weight_decay=1e-4)
fit(3, opt, None, "head only")

# ---------- 4. Phase 2: unfreeze everything, small learning rate ----------
# Low LR so we adjust pretrained knowledge gently instead of destroying it
for p in model.features.parameters():
    p.requires_grad = True
EPOCHS2 = 6
opt = optim.AdamW(model.parameters(), lr=3e-4, weight_decay=1e-4)
sched = optim.lr_scheduler.OneCycleLR(opt, max_lr=3e-4, steps_per_epoch=len(train_loader), epochs=EPOCHS2)
fit(EPOCHS2, opt, sched, "fine-tune")

model.load_state_dict(best_state)
torch.save({"state": model.state_dict(), "classes": CLASSES}, "pets_effnet_b0.pt")

fig, ax = plt.subplots(1, 2, figsize=(11, 4))
ax[0].plot(history["tr_loss"], label="train"); ax[0].plot(history["va_loss"], label="val")
ax[0].axvline(2.5, ls="--", c="gray"); ax[0].set_title("Loss (dashed = unfreeze)"); ax[0].legend()
ax[1].plot(history["tr_acc"], label="train"); ax[1].plot(history["va_acc"], label="val")
ax[1].axvline(2.5, ls="--", c="gray"); ax[1].set_title("Accuracy"); ax[1].legend()
plt.savefig("pets_curves.png", dpi=120); plt.show()

# ---------- 5. Test evaluation (once) ----------
model.eval()
preds, labels = [], []
with torch.no_grad():
    for x, y in test_loader:
        preds += model(x.to(device)).argmax(1).cpu().tolist()
        labels += y.tolist()
preds, labels = np.array(preds), np.array(labels)
print(f"\nTEST ACCURACY: {(preds == labels).mean():.4f}")

# Which breeds get confused most?
cm = confusion_matrix(labels, preds)
off = cm.copy(); np.fill_diagonal(off, 0)
print("\nMost confused breed pairs (true -> predicted : count)")
for idx in np.dstack(np.unravel_index(np.argsort(-off.ravel())[:8], off.shape))[0]:
    t, p = idx
    print(f"  {CLASSES[t]} -> {CLASSES[p]} : {off[t, p]}")

# ---------- 6. Predict on your own photo ----------
def predict(path, top_k=3):
    img = Image.open(path).convert("RGB")
    x = eval_tf(img).unsqueeze(0).to(device)
    model.eval()
    with torch.no_grad():
        probs = torch.softmax(model(x), dim=1)[0]
    top = probs.topk(top_k)
    return [(CLASSES[i], round(p.item() * 100, 1)) for p, i in zip(top.values, top.indices)]

# Example: print(predict("my_cat.jpg"))
