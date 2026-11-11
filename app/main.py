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
    # PaddleOCR 2.7.3 supports the Italian multilingual model and runs on the
    # older CPU available on the home ThinkCentre server.
    logger.info("PaddleOCR init start model=it")
    ocr = PaddleOCR(
        lang="it",
        use_angle_cls=False,
    )
    logger.info("PaddleOCR init complete model=it")
    return ocr

@app.on_event("startup")
def preload_receipt_ocr() -> None:
    """Load PaddleOCR before the service accepts receipt requests."""
    logger.info("PaddleOCR preload start model=it")
    get_receipt_ocr()
    logger.info("PaddleOCR preload complete model=it")

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
        results = ocr.ocr(temp_path, cls=False) or []
        for page in results:
            for line in page or []:
                if not isinstance(line, (list, tuple)) or len(line) < 2:
                    continue
                box, recognition = line[0], line[1]
                if not isinstance(recognition, (list, tuple)) or not recognition:
                    continue
                value = str(recognition[0]).strip()
                if not value:
                    continue
                score = float(recognition[1]) if len(recognition) > 1 else 0.0
                flat_box = [coordinate for point in (box or []) for coordinate in point]
                rows.append({
                    "text": value,
                    "confidence": round(score, 3),
                    "box": [int(round(float(point))) for point in flat_box],
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
