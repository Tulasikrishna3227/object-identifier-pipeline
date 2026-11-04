"""
Tier 1 pipeline: image -> detect objects -> (breed model for cats/dogs) -> detailed description -> report

Colab (GPU runtime):
!pip install -q -U ultralytics transformers accelerate gradio
Optional: upload pets_effnet_b0.pt (from the transfer-learning step) to the Colab file panel.

Stages
 1. YOLO11 detects and counts objects (COCO's 80 classes: person, car, laptop, cat, dog, ...)
 2. Cats/dogs are routed to YOUR fine-tuned breed model (skipped if the .pt file is missing)
 3. Qwen2-VL-2B writes a detailed description of every cropped object
 4. Everything is merged into one report + annotated image
"""
import os, re
from collections import Counter

import torch, torch.nn as nn
import torchvision, torchvision.transforms as T
from PIL import Image, ImageDraw
from ultralytics import YOLO
from transformers import Qwen2VLForConditionalGeneration, AutoProcessor
import gradio as gr

device = "cuda" if torch.cuda.is_available() else "cpu"
print("Using:", device)

# ---------- Stage 1: detector ----------
detector = YOLO("yolo11m.pt")          # downloads automatically; use yolo11n.pt for faster, weaker
CONF = 0.35                            # ignore detections below 35% confidence
LOW_CONF = 0.60                        # below this the detector label may be wrong (e.g. a deer called a sheep)

# ---------- Stage 2: your breed model (optional) ----------
CAT_BREEDS = {"abyssinian", "bengal", "birman", "bombay", "british_shorthair", "egyptian_mau",
              "maine_coon", "persian", "ragdoll", "russian_blue", "siamese", "sphynx"}
BREED_PATH = "pets_effnet_b0.pt"
BREED_THRESHOLD = 70                   # below this % we say "breed uncertain"
breed_model, BREEDS = None, []
if os.path.exists(BREED_PATH):
    ckpt = torch.load(BREED_PATH, map_location="cpu")
    BREEDS = ckpt["classes"]
    breed_model = torchvision.models.efficientnet_b0(weights=None)
    breed_model.classifier[1] = nn.Linear(breed_model.classifier[1].in_features, len(BREEDS))
    breed_model.load_state_dict(ckpt["state"])
    breed_model = breed_model.to(device).eval()
    cat_idx = [i for i, c in enumerate(BREEDS) if c.lower() in CAT_BREEDS]
    dog_idx = [i for i in range(len(BREEDS)) if i not in cat_idx]
    print("Breed model loaded.")
else:
    print(f"{BREED_PATH} not found: breed identification skipped.")

breed_tf = T.Compose([T.Resize(256), T.CenterCrop(224), T.ToTensor(),
                      T.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225))])

def identify_breed(crop, coco_label):
    """Top-3 breeds, restricted to cat breeds for a cat and dog breeds for a dog."""
    x = breed_tf(crop).unsqueeze(0).to(device)
    with torch.no_grad():
        logits = breed_model(x)[0]
        allowed = cat_idx if coco_label == "cat" else dog_idx
        probs = torch.full_like(logits, float("-inf"))
        probs[allowed] = logits[allowed]
        probs = torch.softmax(probs, dim=0)
    top = probs.topk(3)
    return [(BREEDS[i].replace("_", " "), round(p.item() * 100, 1)) for p, i in zip(top.values, top.indices)]

# ---------- Stage 2b: your car make/model model (optional) ----------
CAR_PATH = "cars_effnet_b2.pt"
CAR_THRESHOLD = 40                     # 196 classes, so confidences are naturally lower than for breeds
car_model, CARS = None, []
if os.path.exists(CAR_PATH):
    ckpt = torch.load(CAR_PATH, map_location="cpu")
    CARS = ckpt["classes"]
    car_model = torchvision.models.efficientnet_b2(weights=None)
    car_model.classifier[1] = nn.Linear(car_model.classifier[1].in_features, len(CARS))
    car_model.load_state_dict(ckpt["state"])
    car_model = car_model.to(device).eval()
    print("Car model loaded.")
else:
    print(f"{CAR_PATH} not found: car make/model identification skipped.")

car_tf = T.Compose([T.Resize(292), T.CenterCrop(260), T.ToTensor(),
                    T.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225))])

def identify_car(crop):
    x = car_tf(crop).unsqueeze(0).to(device)
    with torch.no_grad():
        probs = torch.softmax(car_model(x)[0], dim=0)
    top = probs.topk(3)
    return [(CARS[i], round(p.item() * 100, 1)) for p, i in zip(top.values, top.indices)]

# ---------- Stage 3: vision-language model for descriptions ----------
VLM_ID = "Qwen/Qwen2-VL-2B-Instruct"
vlm = Qwen2VLForConditionalGeneration.from_pretrained(
    VLM_ID, torch_dtype=torch.float16, device_map=device, low_cpu_mem_usage=True).eval()   # load straight onto the GPU
processor = AutoProcessor.from_pretrained(VLM_ID)

def prep_crop(img, box, pad=0.08):
    """Crop with a little padding, then keep the size in a range the VLM handles well."""
    x1, y1, x2, y2 = box
    w, h = x2 - x1, y2 - y1
    crop = img.crop((max(0, x1 - pad * w), max(0, y1 - pad * h),
                     min(img.width, x2 + pad * w), min(img.height, y2 + pad * h)))
    if min(crop.size) < 64:
        s = 64 / min(crop.size)
        crop = crop.resize((int(crop.width * s) + 1, int(crop.height * s) + 1))
    crop.thumbnail((448, 448))
    return crop

def describe(crop, label, low_conf=False):
    extra = ""
    if label == "person":
        extra = " Describe clothing, pose and accessories. Do not guess identity."
    elif label in {"car", "truck", "bus", "motorcycle", "bicycle"}:
        extra = (" Name the make and model ONLY if a badge, logo or very distinctive design makes it clear; "
                 "otherwise do not mention the make or model at all. Describe only what is visible.")
    elif label in {"laptop", "cell phone", "tv", "keyboard", "mouse"}:
        extra = (" Name the brand and model ONLY if a logo or text is clearly visible; "
                 "otherwise do not mention the brand or model at all. Describe only what is visible.")
    prompt = (f"This is a cropped image that contains a {label}. Describe ONLY the {label}, ignoring every other "
              f"object in the image: color, shape, material, notable features and condition.{extra} "
              f"Be factual and do not guess. If you are unsure about a detail, leave it out instead of writing 'unspecified'. Answer in 1-2 short sentences.")
    if low_conf:
        # The detector is unsure, so do NOT tell the VLM what the object is: let it name the object itself.
        prompt = ("This is a cropped image of one object. First say what it is, then describe it: color, shape, "
                  "material, notable features and condition. Be factual and do not guess. If you are unsure "
                  "about a detail, leave it out. Answer in 1-2 short sentences.")
    messages = [{"role": "user", "content": [{"type": "image"}, {"type": "text", "text": prompt}]}]
    text = processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    inputs = processor(text=[text], images=[crop], return_tensors="pt").to(device)
    with torch.no_grad():
        out = vlm.generate(**inputs, max_new_tokens=80, do_sample=False)
    out = out[:, inputs["input_ids"].shape[1]:]
    text = processor.batch_decode(out, skip_special_tokens=True)[0].strip()
    sentences = re.split(r"(?<=[.!?])\s+", text)
    return " ".join(sentences[:2])          # the model sometimes ignores "1-2 sentences", so enforce it

# ---------- Stage 4: full pipeline ----------
def analyze(img):
    img = img.convert("RGB")
    res = detector(img, conf=CONF, iou=0.5, verbose=False)[0]      # iou=0.5 merges overlapping boxes of the same class
    boxes = res.boxes.xyxy.cpu().tolist()
    labels = [res.names[int(c)] for c in res.boxes.cls.cpu().tolist()]
    confs = res.boxes.conf.cpu().tolist()
    if not boxes:
        return img, "No objects detected. Try a clearer photo or lower CONF."

    annotated = img.copy()
    draw = ImageDraw.Draw(annotated)
    counts = Counter(labels)
    summary = ", ".join(f"{n} x {l}" for l, n in counts.items())
    report = [f"## Found {len(boxes)} object(s): {summary}\n"]

    for k, (box, label, conf) in enumerate(zip(boxes, labels, confs), 1):
        draw.rectangle(box, outline="red", width=4)
        draw.text((box[0] + 6, box[1] + 4), f"{k}. {label}", fill="red")
        crop = prep_crop(img, box)

        line = f"### {k}. {label.title()}  (detector confidence {conf * 100:.0f}%)\n"
        if conf < LOW_CONF:
            line += f"- **Note:** low detector confidence, so the label '{label}' may be wrong. The description below does not rely on it.\n"
        if breed_model is not None and label in {"cat", "dog"}:
            top = identify_breed(crop, label)
            if top[0][1] >= BREED_THRESHOLD:
                alts = ", ".join(f"{b} {p}%" for b, p in top[1:])
                line += f"- **Breed:** {top[0][0]} ({top[0][1]}%). Next guesses: {alts}\n"
            else:
                line += f"- **Breed:** uncertain, possibly mixed (best guess {top[0][0]}, {top[0][1]}%)\n"
        if car_model is not None and label == "car":
            top = identify_car(crop)
            if top[0][1] >= CAR_THRESHOLD:
                alts = ", ".join(f"{n} {p}%" for n, p in top[1:])
                line += f"- **Make/model (my trained classifier):** {top[0][0]} ({top[0][1]}%). Next guesses: {alts}\n"
            else:
                line += f"- **Make/model:** uncertain (best guess {top[0][0]}, {top[0][1]}%). Interior or partial views are unreliable.\n"
        line += f"- **Description:** {describe(crop, label, conf < LOW_CONF)}\n"
        report.append(line)
    return annotated, "\n".join(report)

# ---------- UI ----------
demo = gr.Interface(
    fn=analyze,
    inputs=gr.Image(type="pil", label="Upload an image"),
    outputs=[gr.Image(label="Detections"), gr.Markdown(label="Report")],
    title="Object identifier: detect, count, identify, describe",
)
if __name__ == "__main__":
    demo.launch(share=True, debug=True)    # share=True gives a public link in Colab
