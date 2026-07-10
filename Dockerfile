# FraudShield API — production container image.
#
# Build:  docker build -t fraudshield-api .
# Run:    docker run -p 8000:8000 -e GROQ_API_KEY=gsk_... fraudshield-api
#
# The image trains the model at BUILD time (baking api/artifacts/ into the
# image), so container startup is instant and doesn't depend on training
# succeeding at runtime. Rebuild the image to pick up model/data changes.

FROM python:3.11-slim

WORKDIR /app

# System deps needed by xgboost/scipy wheels on slim images
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    && rm -rf /var/lib/apt/lists/*

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY . .

# Bake the model artifacts into the image at build time
RUN python -m src.train_pipeline

EXPOSE 8000

ENV PORT=8000

CMD ["sh", "-c", "uvicorn api.app:app --host 0.0.0.0 --port ${PORT}"]
