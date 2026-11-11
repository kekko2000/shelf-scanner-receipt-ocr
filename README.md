# Shelf Food Vision

Servizio interno unificato per il pre-rilevamento degli ingredienti nelle foto del frigorifero e l'OCR locale degli scontrini.

## Flusso

```text
foto frigo → YOLOv8n detector → Shelf
foto scontrino → PaddleOCR → parser Shelf → revisione utente
```

Il servizio non contiene logica Shelf e non sostituisce il Gateway condiviso. Restituisce candidati frigo o righe OCR in JSON. Non usa Gemma né API OCR cloud.

Entrambi gli endpoint usano `AISERVER_API_KEY`, condivisibile con Gateway e llama-server, tramite `Authorization: Bearer ...`.

## Avvio locale

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
uvicorn app.main:app --host 0.0.0.0 --port 8090
```

Su Windows:

```powershell
python -m venv .venv
.venv\Scripts\Activate.ps1
pip install -r requirements.txt
uvicorn app.main:app --host 0.0.0.0 --port 8090
```

## Modello

La build scarica un solo checkpoint: `/models/fridge-ingredient-detector.pt`, dedicato allo scanner frigo.

I checkpoint vengono scaricati durante il build Docker; non vengono duplicati nel nostro repository GitHub.

## Contratto

`POST /detect-fridge` con multipart `file` restituisce:

```json
{
  "candidates": [
    { "name": "apple", "confidence": 0.91, "box": [10, 20, 100, 140] }
  ],
  "model": "fridge-scanner-detector.pt",
  "mode": "fridge"
}
```

La configurazione usa il prefisso `SHELF_SCAN_`:

- `SHELF_SCAN_MODEL_PATH` (predefinito `/models/fridge-ingredient-detector.pt`)
- `SHELF_SCAN_CONFIDENCE` (predefinito `0.35`)
- `SHELF_SCAN_MAX_CANDIDATES` (predefinito `24`)
- `SHELF_SCAN_IMAGE_SIZE` (predefinito `640`)

### OCR scontrino

`POST /receipt` con multipart `file` e campo opzionale `language` (`it` predefinito, `en` supportato) restituisce:

```json
{
  "language": "it",
  "lines": [
    { "text": "LATTE INTERO 1L", "confidence": 0.97, "box": [10, 20, 300, 45] }
  ],
  "mode": "receipt-ocr"
}
```

## Modello scelto

Il training di riferimento è [Rouriqwq123/fridge-ingredient-detector](https://github.com/Rouriqwq123/fridge-ingredient-detector): YOLOv8n fine-tuned su immagini di frigoriferi e ingredienti. Il repository dichiara 88 categorie e non riconosce marche tramite OCR; il servizio restituisce soltanto candidati ingredienti.

## Fine-tuning frigorifero

Il modello specializzato deve essere addestrato su immagini annotate di frigoriferi. Il template è in `training/data.yaml.example` e lo script in `training/train.py`.

```bash
python training/train.py --data /data/fridge-ingredients/data.yaml --weights /models/fridge-ingredient-detector.pt --device cpu
```

Il checkpoint risultante può essere montato nel container e indicato con `SHELF_SCAN_MODEL_PATH`.
