"""
Small object classifier: CNN trained from scratch on CIFAR-10.
Run in Google Colab (Runtime > Change runtime type > GPU), or locally.

pip install torch torchvision matplotlib scikit-learn pillow
Steps covered: data -> split -> augment -> model -> train -> evaluate -> predict
"""
import torch, torch.nn as nn, torch.optim as optim
import torchvision, torchvision.transforms as T
from torch.utils.data import DataLoader, random_split
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix, classification_report
import numpy as np
from PIL import Image

# ---------- 0. Setup ----------
torch.manual_seed(42)
device = "cuda" if torch.cuda.is_available() else "cpu"
print("Using:", device)

CLASSES = ["airplane", "car", "bird", "cat", "deer",
           "dog", "frog", "horse", "ship", "truck"]
MEAN, STD = (0.4914, 0.4822, 0.4465), (0.2470, 0.2435, 0.2616)

# ---------- 1. Data ----------
# Augmentation only on training data. Val/test get plain normalization.
train_tf = T.Compose([
    T.RandomCrop(32, padding=4),
    T.RandomHorizontalFlip(),
    T.ToTensor(),
    T.Normalize(MEAN, STD),
])
eval_tf = T.Compose([T.ToTensor(), T.Normalize(MEAN, STD)])

full_train = torchvision.datasets.CIFAR10("./data", train=True, download=True, transform=train_tf)
full_train_noaug = torchvision.datasets.CIFAR10("./data", train=True, download=True, transform=eval_tf)
test_set = torchvision.datasets.CIFAR10("./data", train=False, download=True, transform=eval_tf)

# Split 45k train / 5k validation (validation = for tuning, test = touched once at the end)
train_idx, val_idx = random_split(range(50000), [45000, 5000],
                                  generator=torch.Generator().manual_seed(42))
train_set = torch.utils.data.Subset(full_train, train_idx.indices)
val_set = torch.utils.data.Subset(full_train_noaug, val_idx.indices)

train_loader = DataLoader(train_set, batch_size=128, shuffle=True, num_workers=2)
val_loader = DataLoader(val_set, batch_size=256, num_workers=2)
test_loader = DataLoader(test_set, batch_size=256, num_workers=2)

# ---------- 2. Model ----------
def block(c_in, c_out):
    # conv finds patterns, BatchNorm stabilizes training, ReLU adds non-linearity, pool shrinks the image
    return nn.Sequential(
        nn.Conv2d(c_in, c_out, 3, padding=1), nn.BatchNorm2d(c_out), nn.ReLU(),
        nn.Conv2d(c_out, c_out, 3, padding=1), nn.BatchNorm2d(c_out), nn.ReLU(),
        nn.MaxPool2d(2),
    )

class SmallCNN(nn.Module):
    def __init__(self, n_classes=10):
        super().__init__()
        self.features = nn.Sequential(block(3, 32), block(32, 64), block(64, 128))  # 32->16->8->4
        self.classifier = nn.Sequential(
            nn.Flatten(), nn.Dropout(0.3),           # dropout reduces overfitting
            nn.Linear(128 * 4 * 4, 256), nn.ReLU(),
            nn.Linear(256, n_classes),               # one score per class
        )
    def forward(self, x):
        return self.classifier(self.features(x))

model = SmallCNN().to(device)
print("Parameters:", sum(p.numel() for p in model.parameters()))

# ---------- 3. Training ----------
EPOCHS = 20
criterion = nn.CrossEntropyLoss()                         # loss for multi-class classification
optimizer = optim.AdamW(model.parameters(), lr=1e-3, weight_decay=1e-4)
scheduler = optim.lr_scheduler.OneCycleLR(optimizer, max_lr=3e-3,
                                          steps_per_epoch=len(train_loader), epochs=EPOCHS)

def run_epoch(loader, train):
    model.train(train)
    total_loss, correct, n = 0, 0, 0
    with torch.set_grad_enabled(train):
        for x, y in loader:
            x, y = x.to(device), y.to(device)
            out = model(x)
            loss = criterion(out, y)
            if train:
                optimizer.zero_grad()
                loss.backward()                           # compute gradients
                optimizer.step()                          # update weights
                scheduler.step()
            total_loss += loss.item() * len(y)
            correct += (out.argmax(1) == y).sum().item()
            n += len(y)
    return total_loss / n, correct / n

history = {"tr_loss": [], "va_loss": [], "tr_acc": [], "va_acc": []}
best_val, best_state = 0, None
for ep in range(EPOCHS):
    tl, ta = run_epoch(train_loader, True)
    vl, va = run_epoch(val_loader, False)
    for k, v in zip(history, (tl, vl, ta, va)):
        history[k].append(v)
    if va > best_val:
        best_val, best_state = va, {k: v.clone() for k, v in model.state_dict().items()}
    print(f"Epoch {ep+1:02d}/{EPOCHS}  train loss {tl:.3f} acc {ta:.3f} | val loss {vl:.3f} acc {va:.3f}")

model.load_state_dict(best_state)                         # keep the best epoch
torch.save(model.state_dict(), "small_cnn_cifar10.pt")

# Learning curves: if train acc >> val acc, the model is overfitting
fig, ax = plt.subplots(1, 2, figsize=(11, 4))
ax[0].plot(history["tr_loss"], label="train"); ax[0].plot(history["va_loss"], label="val")
ax[0].set_title("Loss"); ax[0].legend()
ax[1].plot(history["tr_acc"], label="train"); ax[1].plot(history["va_acc"], label="val")
ax[1].set_title("Accuracy"); ax[1].legend()
plt.savefig("curves.png", dpi=120); plt.show()

# ---------- 4. Final evaluation (test set, used once) ----------
model.eval()
preds, labels = [], []
with torch.no_grad():
    for x, y in test_loader:
        preds += model(x.to(device)).argmax(1).cpu().tolist()
        labels += y.tolist()
print(f"\nTEST ACCURACY: {np.mean(np.array(preds) == np.array(labels)):.4f}")
print(classification_report(labels, preds, target_names=CLASSES))

cm = confusion_matrix(labels, preds)
plt.figure(figsize=(8, 7))
plt.imshow(cm, cmap="Blues"); plt.colorbar()
plt.xticks(range(10), CLASSES, rotation=45); plt.yticks(range(10), CLASSES)
plt.xlabel("Predicted"); plt.ylabel("True"); plt.title("Confusion matrix")
for i in range(10):
    for j in range(10):
        plt.text(j, i, cm[i, j], ha="center", va="center", fontsize=7)
plt.tight_layout(); plt.savefig("confusion_matrix.png", dpi=120); plt.show()

# ---------- 5. Predict on your own image ----------
def predict(path, top_k=3):
    img = Image.open(path).convert("RGB").resize((32, 32))
    x = eval_tf(img).unsqueeze(0).to(device)
    model.eval()
    with torch.no_grad():
        probs = torch.softmax(model(x), dim=1)[0]
    top = probs.topk(top_k)
    return [(CLASSES[i], round(p.item() * 100, 1)) for p, i in zip(top.values, top.indices)]

# Example: print(predict("my_cat.jpg"))
