"""
AmazonNet - Backend FastAPI.
Expone /health y /predict (clasificación multi-etiqueta + Grad-CAM).
"""

import logging
from contextlib import asynccontextmanager

from fastapi import FastAPI, File, HTTPException, UploadFile
from fastapi.middleware.cors import CORSMiddleware

import model as model_module

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("amazonnet")

ALLOWED_CONTENT_TYPES = {"image/jpeg", "image/png", "image/jpg", "image/tiff"}


@asynccontextmanager
async def lifespan(app: FastAPI):
    logger.info("Cargando modelo desde HuggingFace Hub (repo=%s)...", model_module.HF_REPO_ID)
    model_module.load_model()
    logger.info("Modelo cargado correctamente.")
    yield
    logger.info("Apagando backend.")


app = FastAPI(title="AmazonNet API", lifespan=lifespan)

# CORS abierto: el frontend (Streamlit Cloud) llama desde otro dominio.
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.get("/health")
async def health():
    model_loaded = True
    try:
        model_module.get_model()
    except RuntimeError:
        model_loaded = False
    return {"status": "ok", "model_loaded": model_loaded}


@app.post("/predict")
async def predict(file: UploadFile = File(...)):
    if file.content_type not in ALLOWED_CONTENT_TYPES:
        raise HTTPException(
            status_code=400,
            detail=f"Tipo de archivo no soportado: {file.content_type}. "
                   f"Usa JPEG, PNG o TIFF.",
        )

    image_bytes = await file.read()
    if not image_bytes:
        raise HTTPException(status_code=400, detail="El archivo está vacío.")

    try:
        result = model_module.predict(image_bytes)
    except Exception as exc:
        logger.exception("Error durante la predicción")
        raise HTTPException(status_code=500, detail=f"Error en la predicción: {exc}") from exc

    return result