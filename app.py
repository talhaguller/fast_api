from fastapi import FastAPI, File, UploadFile, Request, Form
from fastapi.responses import HTMLResponse
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates
import shutil
import os
import uuid
from model import process_logo

app = FastAPI()

# Statik dosyaları bağla
app.mount("/static", StaticFiles(directory="static"), name="static")
app.mount("/logos", StaticFiles(directory="logos"), name="logos")
app.mount("/uploads", StaticFiles(directory="uploads"), name="uploads")

# uploads klasörünü oluştur (eğer yoksa)
if not os.path.exists("uploads"):
    os.makedirs("uploads")

# Şablonları bağla
templates = Jinja2Templates(directory="templates")

@app.get("/", response_class=HTMLResponse)
async def read_root(request: Request):
    return templates.TemplateResponse("index.html", {"request": request})

@app.post("/upload/")
async def upload_file(
    file: UploadFile = File(...),
    threshold: float = Form(80.0)  # Varsayılan olarak %80
):
    # Benzersiz bir dosya adı oluştur
    unique_filename = f"{uuid.uuid4()}_{file.filename}"
    temp_path = os.path.join("uploads", unique_filename)
    
    # Dosyayı uploads klasörüne kaydet
    with open(temp_path, "wb") as buffer:
        shutil.copyfileobj(file.file, buffer)
    
    # Eşik değerini yüzde formatından 0-1 aralığına çevir
    threshold_normalized = threshold / 100.0
    
    # Logo benzerlik işlemini yap
    logo_folder = "logos"
    results = process_logo(temp_path, logo_folder, threshold=threshold_normalized)
    
    if isinstance(results, dict) and "error" in results:
        os.remove(temp_path)
        return {"error": results["error"]}
    
    # Sonuçları JSON'a uygun hale getir ve test logosunun URL'sini ekle
    formatted_results = [
        {"similarity": float(sim), "path": f"/logos/{os.path.basename(path)}"} 
        for sim, path in results
    ]
    response = {
        "results": formatted_results,
        "test_logo_url": f"/uploads/{unique_filename}"
    }
    
    return response