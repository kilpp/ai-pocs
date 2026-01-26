# Sentiment Analysis API

Multilingual sentiment analysis API built with FastAPI and Hugging Face Transformers. It serves a pre-trained model out of the box and lets you fine-tune a custom model for your domain.

## Features
- REST API with single and batch sentiment endpoints
- Multilingual support via default `cardiffnlp/twitter-xlm-roberta-base-sentiment`
- Pluggable model path (Hugging Face hub id or local fine-tuned directory)
- Minimal training helper for custom datasets
- Dockerfile for containerized deployment

## Quickstart
1. Python 3.11 recommended. Create a virtual environment:
   ```bash
   python -m venv .venv
   source .venv/bin/activate
   ```
2. Install dependencies:
   ```bash
   pip install --upgrade pip
   pip install -r requirements.txt
   ```
3. Run the API (downloads the model on first start):
   ```bash
   uvicorn app.main:app --reload --port 8000
   ```
4. Open docs at http://localhost:8000/docs.

## Configuration
Environment variables (prefixed with `SA_`):
- `SA_MODEL_NAME` (default `cardiffnlp/twitter-xlm-roberta-base-sentiment`)
- `SA_MODEL_REVISION` (optional)
- `SA_DEVICE` (`-1` CPU, `0` for first GPU)
- `SA_PIPELINE_TASK` (default `text-classification`)
- `SA_MAX_LENGTH` (default `512`)
- `SA_BATCH_SIZE` (default `8`)

Example: `SA_MODEL_NAME=your-org/custom-sentiment uvicorn app.main:app --port 8000`

## Example requests
Single:
```bash
curl -X POST http://localhost:8000/sentiment \
  -H "Content-Type: application/json" \
  -d '{"text":"C'est une journée incroyable!","language":"fr"}'
```
Batch:
```bash
curl -X POST http://localhost:8000/sentiment/batch \
  -H "Content-Type: application/json" \
  -d '{"texts":["Me encanta esto","No me gusta"],"language":"es"}'
```

## Custom fine-tuning (optional)
Use the helper script to fine-tune on your CSV dataset:
```bash
python -m app.training.train \
  --train_csv data/train.csv \
  --model_name distilbert-base-multilingual-cased \
  --output_dir model-out \
  --labels '{"negative":0,"neutral":1,"positive":2}' \
  --epochs 2 --batch_size 8
```
Then point the API to the saved model:
```bash
SA_MODEL_NAME=./model-out uvicorn app.main:app --port 8000
```

## Docker
Build and run:
```bash
docker build -t sentiment-api .
docker run -p 8000:8000 --env SA_MODEL_NAME=cardiffnlp/twitter-xlm-roberta-base-sentiment sentiment-api
```

## Tests
Run fast, model-free API tests (uses a fake service override):
```bash
pytest
```
