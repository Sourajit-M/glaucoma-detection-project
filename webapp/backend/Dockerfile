FROM python:3.11-slim

WORKDIR /app

# Layer 1: dependencies (cached until requirements.txt changes)
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Layer 2: application code
COPY app/ ./app/
COPY data/ ./data/

# Layer 3: model files (large, copy last)
COPY models/ ./models/

EXPOSE 8000

CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8000"]