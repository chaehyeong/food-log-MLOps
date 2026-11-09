"""교체 가능한 음식 분류기. transformers가 있으면 Food-101, 없으면 안전한 로컬 폴백."""
from pathlib import Path
import os
from PIL import Image, ImageOps
MODEL_VERSION = "food101-resnet50-optional / fallback-v1"
_pipe = None
MODEL_ID = os.getenv('FOOD_MODEL_ID', 'destinykim679/food-log-mlops-swin-food101')
MODEL_REVISION = os.getenv('FOOD_MODEL_REVISION', 'd38f4f74b4b5d1a9de3e74161b84486ea7e9db8e')

def classify_tta(path: Path):
    """Low-cost horizontal-flip TTA used only for the controlled comparison."""
    global _pipe
    try:
        if _pipe is None:
            from transformers import pipeline
            _pipe = pipeline("image-classification", model=MODEL_ID, revision=MODEL_REVISION)
        with Image.open(path) as image:
            outputs = _pipe(image.convert("RGB"), top_k=3)
            outputs += _pipe(ImageOps.mirror(image.convert("RGB")), top_k=3)
        scores = {}
        for item in outputs:
            scores[item["label"]] = scores.get(item["label"], 0.0) + float(item["score"])
        ranked = sorted(scores.items(), key=lambda item: item[1], reverse=True)[:3]
        predictions = [{"label": label, "confidence": round(score / 2, 4)} for label, score in ranked]
        return {"predictions": predictions, "confidence": predictions[0]["confidence"],
                "model_version": MODEL_ID + "@" + MODEL_REVISION + "+horizontal-flip-tta"}
    except ImportError:
        return classify(path)

def classify_multicrop(path: Path):
    """Three-view crop/resize voting for food occupying different frame areas."""
    global _pipe
    try:
        if _pipe is None:
            from transformers import pipeline
            _pipe = pipeline("image-classification", model=MODEL_ID, revision=MODEL_REVISION)
        with Image.open(path) as image:
            image = image.convert("RGB")
            w, h = image.size
            side = min(w, h)
            left, top = (w - side) // 2, (h - side) // 2
            views = [image, image.crop((left, top, left + side, top + side)),
                     ImageOps.contain(image, (side, side))]
            outputs = [item for view in views for item in _pipe(view, top_k=3)]
        scores = {}
        for item in outputs:
            scores[item["label"]] = scores.get(item["label"], 0.0) + float(item["score"])
        ranked = sorted(scores.items(), key=lambda item: item[1], reverse=True)[:3]
        predictions = [{"label": label, "confidence": round(score / 3, 4)} for label, score in ranked]
        return {"predictions": predictions, "confidence": predictions[0]["confidence"],
                "model_version": MODEL_ID + "@" + MODEL_REVISION + "+multicrop"}
    except ImportError:
        return classify(path)
def classify(path: Path):
    global _pipe
    try:
        if _pipe is None:
            from transformers import pipeline
            _pipe = pipeline("image-classification", model=MODEL_ID, revision=MODEL_REVISION)
        out = _pipe(str(path), top_k=3)
        return {"predictions": [{"label": x["label"], "confidence": round(float(x["score"]), 4)} for x in out], "confidence": float(out[0]["score"]), "model_version": MODEL_ID + "@" + MODEL_REVISION}
    except ImportError:
        pass
    # 실행은 보장하되 추정임을 명확히 표시하는 휴리스틱 폴백
    return {"predictions": [{"label": "음식 (직접 선택)", "confidence": 0.0}, {"label": "밥/면 요리", "confidence": 0.0}, {"label": "국/찌개", "confidence": 0.0}], "confidence": 0.0, "model_version": MODEL_VERSION}
