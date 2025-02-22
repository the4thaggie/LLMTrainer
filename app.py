from fastapi import FastAPI, Depends, HTTPException, Request, UploadFile, File
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates
from fastapi.responses import HTMLResponse, JSONResponse, FileResponse
from src.training_engine import TrainingEngine
from src.model_manager import ModelRegistry
from src.utils.logger import setup_logger
from config import Config
import shutil
import os

app = FastAPI(title="AI Training System")
config = Config()
logger = setup_logger()
model_registry = ModelRegistry()
training_engine = TrainingEngine(config)

# Mount static files
app.mount("/static", StaticFiles(directory="gui/static"), name="static")
templates = Jinja2Templates(directory="gui/templates")

@app.get("/", response_class=HTMLResponse)
async def root(request: Request):
    return templates.TemplateResponse("index.html", {"request": request})

@app.post("/training/jobs")
async def create_training_job(config: dict):
    job_id = training_engine.create_job(config)
    return {"job_id": job_id, "status": "queued"}

@app.get("/training/jobs/{job_id}")
async def get_job_status(job_id: str):
    status = training_engine.get_job_status(job_id)
    return {"job_id": job_id, "status": status}

@app.post("/models/predict")
async def predict(request: dict):
    result = model_registry.predict(request["model_id"], request["input"])
    return result

@app.post("/gui/upload")
async def upload_file(file: UploadFile = File(...)):
    file_path = f"uploads/{file.filename}"
    with open(file_path, "wb") as buffer:
        shutil.copyfileobj(file.file, buffer)
    return {"filename": file.filename, "path": file_path}

@app.get("/gui/download/{model_id}")
async def download_model(model_id: str):
    model_path = model_registry.get_model_path(model_id)
    if not os.path.exists(model_path):
        raise HTTPException(status_code=404, detail="Model not found")
    return FileResponse(model_path, filename=f"{model_id}.pth")

# Add more endpoints as needed

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
