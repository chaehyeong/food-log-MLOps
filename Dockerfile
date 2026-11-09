FROM python:3.11-slim

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    HF_HOME=/opt/huggingface \
    TRANSFORMERS_CACHE=/opt/huggingface/transformers \
    FOOD_MODEL_ID=destinykim679/food-log-mlops-swin-food101 \
    FOOD_MODEL_REVISION=d38f4f74b4b5d1a9de3e74161b84486ea7e9db8e

WORKDIR /app

RUN apt-get update \
    && apt-get install -y --no-install-recommends libgomp1 \
    && rm -rf /var/lib/apt/lists/*

COPY requirements.txt .
RUN python -m pip install --upgrade pip \
    && python -m pip install -r requirements.txt

COPY app.py model.py ./
COPY static ./static
RUN mkdir -p data/uploads /opt/huggingface

# Download the production model while building and verify that it can infer.
RUN python - <<'PY'
from pathlib import Path
from PIL import Image
from model import classify

sample = Path('/tmp/model-smoke.jpg')
Image.new('RGB', (224, 224), (128, 128, 128)).save(sample)
result = classify(sample)
assert result['model_version'].startswith('destinykim679/food-log-mlops-swin-food101@')
assert len(result['predictions']) == 3
print('Model smoke test passed:', result['model_version'])
sample.unlink()
PY

VOLUME ["/app/data"]
EXPOSE 8000
HEALTHCHECK --interval=30s --timeout=5s --start-period=20s --retries=3 \
  CMD python -c "import urllib.request; urllib.request.urlopen('http://127.0.0.1:8000/docs')"

CMD ["uvicorn", "app:app", "--host", "0.0.0.0", "--port", "8000"]
