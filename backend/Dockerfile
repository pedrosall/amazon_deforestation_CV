FROM python:3.11-slim

# Evita que Python genere archivos .pyc y fuerza logs sin buffer (útil para ver logs en tiempo real en Docker)
ENV PYTHONDONTWRITEBYTECODE=1
ENV PYTHONUNBUFFERED=1

# Dependencias del sistema necesarias para opencv-python-headless
RUN apt-get update && apt-get install -y --no-install-recommends \
    libgl1 \
    libglib2.0-0 \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

# Copiamos primero solo requirements.txt para aprovechar el cache de capas de Docker:
# si el código cambia pero no las dependencias, no se reinstala todo desde cero
COPY requirements.txt .

# Instalamos PyTorch en su variante CPU explícitamente (mucho más ligera que la versión con CUDA)
RUN pip install --no-cache-dir torch==2.6.0 torchvision==0.21.0 --index-url https://download.pytorch.org/whl/cpu
RUN pip install --no-cache-dir -r requirements.txt

# Ahora sí copiamos el resto del código
COPY . .

EXPOSE 7860

CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "7860"]