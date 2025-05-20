# Gunakan image base Python 3.11.9 yang ringan
FROM python:3.11.9-slim

# Tentukan direktori kerja di dalam container
WORKDIR /app

# Salin requirements.txt dan install dependensi
COPY requirements.txt .

RUN pip install --upgrade pip && pip install --no-cache-dir -r requirements.txt

# Salin seluruh isi project ke dalam container
COPY . .

# Expose port FastAPI
EXPOSE 8000

# Jalankan aplikasi menggunakan Uvicorn
CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8000"]
