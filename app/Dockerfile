FROM python:3.11-slim

WORKDIR /app
COPY requirements.txt requirements.txt


RUN apt-get update && \
    apt-get install -y \
    build-essential \
    python3-dev \
    python3-setuptools \
    tesseract-ocr \
    libgl1 \
    make \
    gcc \
    libmagic1 \
    poppler-utils \
    unrtf \
    build-essential \
    libpoppler-cpp-dev \
    pkg-config 

RUN pip install --no-cache-dir --prefer-binary -r requirements.txt

COPY . .
EXPOSE 5003

CMD ["uvicorn", "endpoint_ocr:app", "--host", "0.0.0.0", "--port", "5003"]
