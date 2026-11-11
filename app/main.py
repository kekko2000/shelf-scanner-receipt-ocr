import json
import logging
import os
import secrets
from functools import lru_cache
from pathlib import Path
from tempfile import NamedTemporaryFile

from fastapi import FastAPI, File, Form, Header, HTTPException, UploadFile
from paddleocr import PaddleOCR
from PIL import Image

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("shelf-scan-receipt")

app = FastAPI(title="Shelf Receipt OCR", version="1.0.0")
DETECTOR_API_KEY = os.getenv("AISERVER_API_KEY", "").strip()

@lru_cache(maxsize=1)
def get_receipt_ocr() -> PaddleOCR:
    # One PP-OCRv5 Latin model covers both Italian and English receipts.
    logger.info("PaddleOCR init start model=latin")
    ocr = PaddleOCR(
        lang="latin",
        use_doc_orientation_classify=False,
        use_doc_unwarping=False,
        use_textline_orientation=False,
    )
    logger.info("PaddleOCR init complete model=latin")
    return ocr

@app.on_event("startup")
def preload_receipt_ocr() -> None:
    """Load the default Italian OCR model before accepting requests."""
    logger.info("PaddleOCR preload start model=latin")
    get_receipt_ocr()
    logger.info("PaddleOCR preload complete model=latin")

def require_api_key(authorization: str | None) -> None:
    if not DETECTOR_API_KEY:
        raise HTTPException(status_code=503, detail="API key non configurata.")
    raw = (authorization or "").strip()
    token = raw[7:].strip() if raw.lower().startswith("bearer ") else raw
    if not token or not secrets.compare_digest(token, DETECTOR_API_KEY):
        raise HTTPException(status_code=401, detail="API key non valida.")

def validate_image(data: bytes, filename: str) -> str:
    if not data or len(data) > 12 * 1024 * 1024:
        raise HTTPException(status_code=400, detail="Immagine mancante o troppo grande.")
    suffix = Path(filename or "receipt.jpg").suffix or ".jpg"
    with NamedTemporaryFile(suffix=suffix, delete=False) as temp:
        temp.write(data)
        temp_path = temp.name
    try:
        Image.open(temp_path).verify()
    except Exception as exc:
        try:
            os.unlink(temp_path)
        except OSError:
            pass
        raise HTTPException(status_code=400, detail="Immagine non valida.") from exc
    return temp_path

@app.get("/health")
def health() -> dict[str, str]:
    return {"status": "ok", "service": "shelf-scan-receipt"}

@app.get("/")
def root() -> dict[str, str]:
    return {"status": "ok", "service": "shelf-scan-receipt"}

@app.post("/receipt")
async def ocr_receipt(
    file: UploadFile = File(...),
    language: str = Form("it"),
    authorization: str | None = Header(default=None),
) -> dict:
    require_api_key(authorization)
    if not file.content_type or not file.content_type.startswith("image/"):
        raise HTTPException(status_code=400, detail="Il file deve essere un'immagine.")
    temp_path = validate_image(await file.read(), file.filename or "receipt.jpg")
    selected_language = "en" if language.lower().startswith("en") else "it"
    try:
        logger.info("OCR request start language=%s filename=%s", selected_language, file.filename)
        rows: list[dict] = []
        ocr = get_receipt_ocr()
        logger.info("OCR predict start language=%s", selected_language)
        for result in ocr.predict(temp_path):
            payload = getattr(result, "json", result)
            if callable(payload):
                payload = payload()
            if isinstance(payload, str):
                payload = json.loads(payload)
            if not isinstance(payload, dict):
                continue
            texts = payload.get("rec_texts") or []
            scores = payload.get("rec_scores") or []
            boxes = payload.get("rec_boxes") or []
            for index, text in enumerate(texts):
                value = str(text).strip()
                if not value:
                    continue
                score = float(scores[index]) if index < len(scores) else 0.0
                box = boxes[index] if index < len(boxes) else []
                rows.append({
                    "text": value,
                    "confidence": round(score, 3),
                    "box": [int(round(float(point))) for point in box] if box else [],
                })
        rows.sort(key=lambda row: (row["box"][1] if len(row["box"]) > 1 else 0, row["box"][0] if row["box"] else 0))
        logger.info("OCR request complete language=%s lines=%s", selected_language, len(rows))
        return {"language": selected_language, "lines": rows, "mode": "receipt-ocr"}
    except Exception as exc:
        logger.exception("OCR request failed language=%s", selected_language)
        raise HTTPException(status_code=422, detail=f"Impossibile leggere lo scontrino: {exc}") from exc
    finally:
        try:
            os.unlink(temp_path)
        except OSError:
            pass
