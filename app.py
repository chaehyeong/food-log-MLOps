from pathlib import Path
import json, sqlite3, uuid
from datetime import date
from fastapi import FastAPI, File, Form, UploadFile, HTTPException
from fastapi.responses import HTMLResponse, FileResponse
from fastapi.staticfiles import StaticFiles
from model import classify

ROOT = Path(__file__).parent
DATA = ROOT / "data"
UPLOADS = DATA / "uploads"
DB = DATA / "foodlog.sqlite3"
UPLOADS.mkdir(parents=True, exist_ok=True)
app = FastAPI(title="오늘 뭐 먹었지?")
app.mount("/static", StaticFiles(directory=ROOT / "static"), name="static")

def conn():
    c = sqlite3.connect(DB); c.row_factory = sqlite3.Row
    c.execute("""CREATE TABLE IF NOT EXISTS meals(
      id INTEGER PRIMARY KEY, image_path TEXT NOT NULL, predictions TEXT NOT NULL,
      confidence REAL NOT NULL, model_version TEXT NOT NULL, selected_label TEXT,
      eaten_date TEXT NOT NULL, comment TEXT DEFAULT '', created_at TEXT DEFAULT CURRENT_TIMESTAMP)""")
    c.commit(); return c

@app.get("/", response_class=HTMLResponse)
def home(): return (ROOT / "static" / "index.html").read_text(encoding="utf-8")

@app.get("/uploads/{name}")
def image(name: str):
    p = UPLOADS / name
    if not p.is_file(): raise HTTPException(404, "이미지를 찾을 수 없습니다")
    return FileResponse(p)

@app.post("/api/meals")
async def upload(image: UploadFile = File(...), eaten_date: str = Form(...)):
    try: date.fromisoformat(eaten_date)
    except ValueError: raise HTTPException(400, "먹은 날짜 형식이 올바르지 않습니다")
    if not image.content_type or not image.content_type.startswith("image/"):
        raise HTTPException(400, "이미지 파일만 업로드할 수 있습니다")
    raw = await image.read()
    if len(raw) > 10 * 1024 * 1024: raise HTTPException(413, "이미지는 10MB 이하만 가능합니다")
    ext = Path(image.filename or "photo.jpg").suffix.lower() or ".jpg"
    if ext not in {".jpg", ".jpeg", ".png", ".webp"}: ext = ".jpg"
    name = uuid.uuid4().hex + ext; (UPLOADS / name).write_bytes(raw)
    try: result = classify(UPLOADS / name)
    except Exception as e: (UPLOADS / name).unlink(missing_ok=True); raise HTTPException(503, f"모델 분석 오류: {e}")
    c = conn(); cur = c.execute("INSERT INTO meals(image_path,predictions,confidence,model_version,eaten_date) VALUES(?,?,?,?,?)",
      (name, json.dumps(result["predictions"], ensure_ascii=False), result["confidence"], result["model_version"], eaten_date)); c.commit()
    return {"id": cur.lastrowid, "image_url": f"/uploads/{name}", **result, "eaten_date": eaten_date}

@app.get("/api/meals")
def meals(year: int | None = None, month: int | None = None):
    c = conn(); rows = c.execute("SELECT * FROM meals ORDER BY eaten_date, id").fetchall()
    return [{**dict(r), "predictions": json.loads(r["predictions"]), "image_url": f"/uploads/{r['image_path']}"} for r in rows]

@app.patch("/api/meals/{meal_id}")
async def update(meal_id: int, payload: dict):
    c = conn(); row = c.execute("SELECT id FROM meals WHERE id=?", (meal_id,)).fetchone()
    if not row: raise HTTPException(404, "기록을 찾을 수 없습니다")
    allowed = {k: payload[k] for k in ("selected_label", "comment") if k in payload}
    if not allowed: raise HTTPException(400, "변경할 값이 없습니다")
    c.execute("UPDATE meals SET " + ",".join(f"{k}=?" for k in allowed) + " WHERE id=?", (*allowed.values(), meal_id)); c.commit()
    return {"ok": True}

@app.delete("/api/meals/{meal_id}")
def delete_meal(meal_id: int):
    c = conn(); row = c.execute("SELECT image_path FROM meals WHERE id=?", (meal_id,)).fetchone()
    if not row: raise HTTPException(404, "기록을 찾을 수 없습니다")
    c.execute("DELETE FROM meals WHERE id=?", (meal_id,)); c.commit()
    (UPLOADS / row["image_path"]).unlink(missing_ok=True)
    return {"ok": True}
