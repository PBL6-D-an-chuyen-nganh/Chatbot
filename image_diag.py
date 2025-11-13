# image_diag.py
import json, os, torch
import torch.nn as nn
from torchvision.models import resnet101

# cache để không tạo lại nhiều lần
_MODEL = None
_DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def _load_labels(labels_path):
    with open(labels_path, "r", encoding="utf-8") as f:
        labels = json.load(f)
    # labels có thể là list hoặc dict {id: name}. Chuẩn hoá về list
    if isinstance(labels, dict):
        # sắp xếp theo key để ổn định
        labels = [labels[k] for k in sorted(labels.keys(), key=lambda x: int(x) if str(x).isdigit() else x)]
    return labels

def _build_model(num_classes: int):
    m = resnet101(weights=None)  # không dùng head 1000 lớp mặc định
    in_feats = m.fc.in_features
    m.fc = nn.Linear(in_feats, num_classes)  # gắn đúng head 7 lớp (ví dụ)
    return m

def _load_checkpoint(m, class_path):
    state = torch.load(class_path, map_location="cpu")
    # nếu checkpoint lưu "state_dict" bên trong, bóc ra:
    if isinstance(state, dict) and "state_dict" in state:
        state = state["state_dict"]

    # Một số lib lưu tên key có prefix 'module.' → bỏ để khớp
    new_state = {}
    for k, v in state.items():
        nk = k.replace("module.", "")  # bỏ DistributedDataParallel prefix
        new_state[nk] = v

    # nạp lỏng để không lỗi nếu có key lạ
    missing, unexpected = m.load_state_dict(new_state, strict=False)
    if missing or unexpected:
        print(f"[image_diag] load_state_dict: missing={missing}, unexpected={unexpected}")
    return m

def _ensure_model(class_path, labels_path):
    global _MODEL
    if _MODEL is not None:
        return _MODEL

    labels = _load_labels(labels_path)
    num_classes = len(labels)

    m = _build_model(num_classes)
    if class_path and os.path.isfile(class_path):
        m = _load_checkpoint(m, class_path)

    m.eval().to(_DEVICE)
    _MODEL = (m, labels)
    return _MODEL

@torch.inference_mode()
def predict_image(raw_bytes: bytes, top_k=3, class_path=None, seg_path=None, labels_path=None):
    """
    raw_bytes: nội dung file ảnh
    class_path: đường dẫn checkpoint resnet101.pth (đã train với số lớp = len(labels))
    labels_path: đường dẫn labels.json
    """
    from PIL import Image
    from io import BytesIO
    from torchvision import transforms as T

    if labels_path is None:
        labels_path = os.path.join(os.path.dirname(__file__), "models", "labels.json")

    model, labels = _ensure_model(class_path, labels_path)

    img = Image.open(BytesIO(raw_bytes)).convert("RGB")
    tfm = T.Compose([
        T.Resize(256),
        T.CenterCrop(224),
        T.ToTensor(),
        T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])
    x = tfm(img).unsqueeze(0).to(_DEVICE)

    logits = model(x)
    probs = torch.softmax(logits, dim=1)[0]
    k = min(top_k, probs.numel())
    topk = torch.topk(probs, k)

    out = []
    for i in range(k):
        idx = topk.indices[i].item()
        out.append({
            "label": labels[idx] if idx < len(labels) else str(idx),
            "score": float(topk.values[i].item()),
        })
    return out
