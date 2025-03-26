# Python 3.12-slim imajını temel al
FROM python:3.12-slim

# Çalışma dizinini ayarla
WORKDIR /app

# Gereksinim dosyasını kopyala
COPY requirements.txt .

# Bağımlılıkları yükle
RUN pip install --no-cache-dir -r requirements.txt

# Proje dosyalarını kopyala
COPY . .

# Uvicorn ile FastAPI'yi çalıştır
CMD ["uvicorn", "app:app", "--host", "0.0.0.0", "--port", "8000"]