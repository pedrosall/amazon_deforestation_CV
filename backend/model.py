"""
AmazonNet - Carga de modelo, preprocesado, inferencia y Grad-CAM.
"""

import base64
import io
import os

import cv2
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from huggingface_hub import hf_hub_download
from PIL import Image
from torchvision import transforms
from torchvision.models import efficientnet_b3

# --------------------------------------------------------------------------
# Configuración
# --------------------------------------------------------------------------

HF_REPO_ID = os.getenv("HF_REPO_ID", "pedrosall/amazonnet-efficientnet-b3")
HF_FILENAME = os.getenv("HF_FILENAME", "best_efficientnet_finetuned.pth")
HF_TOKEN = os.getenv("HF_TOKEN")  # solo necesario si el repo es privado

THRESHOLD = 0.3

ALL_LABELS = sorted([
    "agriculture", "artisinal_mine", "bare_ground", "blooming",
    "blow_down", "clear", "cloudy", "conventional_mine", "cultivation",
    "habitation", "haze", "partly_cloudy", "primary", "road",
    "selective_logging", "slash_burn", "water",
])

THREAT_LABELS = {
    "artisinal_mine", "blow_down", "conventional_mine",
    "selective_logging", "slash_burn",
}

IMAGE_SIZE = 300

PREPROCESS = transforms.Compose([
    transforms.Resize((IMAGE_SIZE, IMAGE_SIZE)),
    transforms.ToTensor(),
    transforms.Normalize(
        mean=[0.485, 0.456, 0.406],
        std=[0.229, 0.224, 0.225],
    ),
])

# --------------------------------------------------------------------------
# Estado global (se rellena en load_model, llamado en el lifespan de FastAPI)
# --------------------------------------------------------------------------

_model: nn.Module | None = None
_gradcam_activations = None
_gradcam_gradients = None


def _save_activation(module, input, output):
    global _gradcam_activations
    _gradcam_activations = output.detach()


def _save_gradient(module, grad_input, grad_output):
    global _gradcam_gradients
    _gradcam_gradients = grad_output[0].detach()


def build_model() -> nn.Module:
    """Reconstruye la arquitectura exacta usada en el finetuning."""
    model = efficientnet_b3(weights=None)
    in_features = model.classifier[1].in_features  # 1536
    model.classifier = nn.Sequential(
        nn.Dropout(p=0.3, inplace=True),
        nn.Linear(in_features, len(ALL_LABELS)),
    )
    return model


def load_model() -> nn.Module:
    """Descarga el checkpoint desde HuggingFace Hub y carga el modelo en memoria."""
    global _model

    checkpoint_path = hf_hub_download(
        repo_id=HF_REPO_ID,
        filename=HF_FILENAME,
        token=HF_TOKEN,
    )

    model = build_model()
    state_dict = torch.load(checkpoint_path, map_location="cpu")
    model.load_state_dict(state_dict)
    model.eval()

    # Registrar hooks de Grad-CAM sobre la última capa conv del feature extractor
    target_layer = model.features[-1]
    target_layer.register_forward_hook(_save_activation)
    target_layer.register_full_backward_hook(_save_gradient)

    _model = model
    return model


def get_model() -> nn.Module:
    if _model is None:
        raise RuntimeError("El modelo no está cargado. Llama a load_model() primero.")
    return _model


# --------------------------------------------------------------------------
# Preprocesado
# --------------------------------------------------------------------------

def preprocess_image(image_bytes: bytes) -> tuple[torch.Tensor, Image.Image]:
    """Devuelve (tensor listo para el modelo, imagen PIL redimensionada para overlay)."""
    image = Image.open(io.BytesIO(image_bytes)).convert("RGB")
    tensor = PREPROCESS(image).unsqueeze(0)  # [1, 3, 300, 300]
    resized_for_overlay = image.resize((IMAGE_SIZE, IMAGE_SIZE))
    return tensor, resized_for_overlay


# --------------------------------------------------------------------------
# Grad-CAM
# --------------------------------------------------------------------------

def compute_gradcam(model: nn.Module, input_tensor: torch.Tensor, class_idx: int) -> np.ndarray:
    """
    Calcula el mapa Grad-CAM para una única clase.
    IMPORTANTE: requiere backward, no debe llamarse dentro de torch.no_grad().
    Devuelve un array [0, 1] de tamaño (IMAGE_SIZE, IMAGE_SIZE).
    """
    model.zero_grad()
    input_tensor = input_tensor.clone().requires_grad_(True)

    output = model(input_tensor)  # [1, 17]
    score = output[0, class_idx]
    score.backward()

    activations = _gradcam_activations[0]  # [C, H, W]
    gradients = _gradcam_gradients[0]      # [C, H, W]

    weights = gradients.mean(dim=(1, 2))   # [C]
    cam = torch.zeros(activations.shape[1:], dtype=torch.float32)
    for i, w in enumerate(weights):
        cam += w * activations[i]

    cam = F.relu(cam)
    cam = cam / (cam.max() + 1e-8)
    cam = cam.numpy()

    cam_resized = cv2.resize(cam, (IMAGE_SIZE, IMAGE_SIZE))
    return cam_resized


def overlay_gradcam(base_image: Image.Image, cam: np.ndarray, alpha: float = 0.45) -> str:
    """Superpone el heatmap Grad-CAM sobre la imagen base y devuelve un PNG en base64."""
    base_rgb = np.array(base_image)  # RGB
    heatmap = np.uint8(255 * cam)
    heatmap_color = cv2.applyColorMap(heatmap, cv2.COLORMAP_JET)  # BGR
    heatmap_color = cv2.cvtColor(heatmap_color, cv2.COLOR_BGR2RGB)

    overlay = np.uint8(base_rgb * (1 - alpha) + heatmap_color * alpha)

    overlay_image = Image.fromarray(overlay)
    buffer = io.BytesIO()
    overlay_image.save(buffer, format="PNG")
    return base64.b64encode(buffer.getvalue()).decode("utf-8")


# --------------------------------------------------------------------------
# Inferencia end-to-end
# --------------------------------------------------------------------------

def predict(image_bytes: bytes) -> dict:
    model = get_model()
    input_tensor, base_image = preprocess_image(image_bytes)

    # Forward sin gradientes para las probabilidades
    with torch.no_grad():
        logits = model(input_tensor)
        probs_tensor = torch.sigmoid(logits)[0]

    probs = {label: float(probs_tensor[i]) for i, label in enumerate(ALL_LABELS)}
    active_labels = [label for label, p in probs.items() if p >= THRESHOLD]
    threat_labels = [label for label in active_labels if label in THREAT_LABELS]
    top_label = max(probs, key=probs.get)
    top_idx = ALL_LABELS.index(top_label)

    # Forward + backward con gradientes solo para la clase top (Grad-CAM)
    cam = compute_gradcam(model, input_tensor, top_idx)
    gradcam_b64 = overlay_gradcam(base_image, cam)

    return {
        "probs": probs,
        "active_labels": active_labels,
        "threat_labels": threat_labels,
        "gradcam_b64": gradcam_b64,
        "top_label": top_label,
    }