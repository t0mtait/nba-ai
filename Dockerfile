FROM python:3.11-slim

WORKDIR /app

# Install kagglehub and other dependencies
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy application and model artifacts
COPY app.py .
COPY models/ ./models/

ENV FLASK_APP=app.py
ENV FLASK_RUN_HOST=0.0.0.0

# Railway injects PORT env var - use it, defaulting to 5000
ENV PORT=5000

EXPOSE ${PORT}

CMD ["flask", "run", "--host=0.0.0.0", "--port=${PORT}"]