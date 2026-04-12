FROM python:3.11-slim

WORKDIR /app

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt \
    && pip install --no-cache-dir "huggingface_hub==0.23.4"

COPY app/ ./app/
COPY data/ ./data/

RUN python - <<'EOF'
from huggingface_hub import hf_hub_download
import shutil, os

repo = "sourajitm19/glaucoma-detect"
os.makedirs("models", exist_ok=True)

for filename in [
    "glaucoma_resnet18.onnx",
    "disc_unet.onnx",
    "cup_unet.onnx",
]:
    print(f"Downloading {filename} ...", flush=True)
    path = hf_hub_download(repo_id=repo, filename=filename)
    shutil.copy(path, f"models/{filename}")
    print(f"  OK -> models/{filename}", flush=True)

print("All models ready.", flush=True)
EOF

EXPOSE 8000
CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8000"]