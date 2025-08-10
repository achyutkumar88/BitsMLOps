# Dockerfile
FROM python:3.9-slim

WORKDIR /app

#COPY . /app

COPY . .

RUN pip install -r requirements.txt

ENV PYTHONPATH=/app

EXPOSE 8000

CMD ["uvicorn", "src.app:app", "--host", "0.0.0.0", "--port", "8000"]