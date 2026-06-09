FROM python:3.11-slim

WORKDIR /app

RUN pip install --no-cache-dir fastapi==0.111.0 "uvicorn[standard]==0.29.0" boto3==1.34.0 anthropic python-multipart

COPY src/dashboard/api.py ./dashboard/api.py
COPY src/dashboard/__init__.py ./dashboard/__init__.py

EXPOSE 8000

CMD ["uvicorn", "dashboard.api:app", "--host", "0.0.0.0", "--port", "8000"]
