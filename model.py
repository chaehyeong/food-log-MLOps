"""교체 가능한 음식 분류기. transformers가 있으면 Food-101, 없으면 안전한 로컬 폴백."""
from pathlib import Path
MODEL_VERSION = "food101-resnet50-optional / fallback-v1"
_pipe = None
def classify(path: Path):
    global _pipe
    try:
        if _pipe is None:
            from transformers import pipeline
            _pipe = pipeline("image-classification", model="nateraw/food")
        out = _pipe(str(path), top_k=3)
        return {"predictions": [{"label": x["label"], "confidence": round(float(x["score"]), 4)} for x in out], "confidence": float(out[0]["score"]), "model_version": "nateraw/food"}
    except ImportError:
        pass
    # 실행은 보장하되 추정임을 명확히 표시하는 휴리스틱 폴백
    return {"predictions": [{"label": "음식 (직접 선택)", "confidence": 0.0}, {"label": "밥/면 요리", "confidence": 0.0}, {"label": "국/찌개", "confidence": 0.0}], "confidence": 0.0, "model_version": MODEL_VERSION}
