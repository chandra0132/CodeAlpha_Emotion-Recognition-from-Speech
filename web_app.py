import os
import csv
import tempfile
import uvicorn
import glob
import random
from fastapi import FastAPI, UploadFile, File, HTTPException
from fastapi.responses import HTMLResponse
from fastapi.staticfiles import StaticFiles
from src.predict import EmotionPredictor
from src.feature_extraction import EMOTION_MAP

app = FastAPI(title="Speech Emotion Recognition API")

# Mount directories to serve files directly
app.mount("/results", StaticFiles(directory="results"), name="results")
app.mount("/data", StaticFiles(directory="data"), name="data")


# Initialize the predictor globally so it loads the model once at startup
predictor = None

@app.on_event("startup")
def load_model():
    global predictor
    try:
        # Resolve absolute path for robustness
        base_dir = os.path.abspath(os.path.dirname(__file__))
        models_dir = os.path.join(base_dir, "results", "models")
        if not os.path.exists(models_dir):
            raise FileNotFoundError(f"Models directory not found at {models_dir}")
        predictor = EmotionPredictor(models_dir=models_dir)
        print("[INFO] Model loaded successfully.")
    except Exception as e:
        print(f"[ERROR] Failed to load the predictor model: {e}")

@app.get("/random_demo_audio")
def get_random_demo_audio():
    try:
        base_dir = os.path.abspath(os.path.dirname(__file__))
        data_dir = os.path.join(base_dir, "data")
        wav_files = glob.glob(os.path.join(data_dir, "Actor_*", "*.wav"))
        
        if not wav_files:
            wav_files = glob.glob(os.path.join(data_dir, "*.wav"))
            
        if not wav_files:
            raise HTTPException(status_code=404, detail="No demo audio files found in data directory.")
            
        selected_file = random.choice(wav_files)
        # Create URL-friendly path
        rel_path = os.path.relpath(selected_file, base_dir)
        url_path = "/" + rel_path.replace(os.sep, "/")
        
        filename = os.path.basename(selected_file)
        parts = filename.split('-')
        ground_truth = "Unknown"
        if len(parts) >= 3:
            code = parts[2]
            if code in EMOTION_MAP:
                ground_truth = EMOTION_MAP[code][0]
                
        return {
            "success": True,
            "file_url": url_path,
            "file_name": filename,
            "ground_truth": ground_truth.capitalize()
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/", response_class=HTMLResponse)
def read_root():
    base_dir = os.path.abspath(os.path.dirname(__file__))
    template_path = os.path.join(base_dir, "templates", "index.html")
    if not os.path.exists(template_path):
        raise HTTPException(status_code=404, detail="Frontend template index.html not found.")
    with open(template_path, "r", encoding="utf-8") as f:
        return f.read()

@app.post("/predict")
async def predict_emotion(file: UploadFile = File(...)):
    global predictor
    if predictor is None:
        raise HTTPException(status_code=503, detail="Prediction model is not initialized/loaded.")
        
    try:
        # Create a temporary file to save the uploaded audio data
        with tempfile.NamedTemporaryFile(delete=False, suffix=".wav") as temp_file:
            content = await file.read()
            temp_file.write(content)
            temp_file_path = temp_file.name
            
        try:
            # Predict emotion
            emotion, confidence, probs = predictor.predict(temp_file_path)
            return {
                "success": True,
                "predicted_emotion": emotion,
                "confidence": float(confidence),
                "probabilities": probs
            }
        finally:
            # Ensure the temporary file is deleted after prediction
            if os.path.exists(temp_file_path):
                os.remove(temp_file_path)
                
    except Exception as e:
        return {
            "success": False,
            "error": str(e)
        }

@app.get("/performance")
def get_performance():
    try:
        base_dir = os.path.abspath(os.path.dirname(__file__))
        csv_path = os.path.join(base_dir, "results", "model_comparison.csv")
        if not os.path.exists(csv_path):
            raise HTTPException(status_code=404, detail="Performance comparison CSV file not found.")
        
        performance_data = []
        with open(csv_path, mode="r", encoding="utf-8") as f:
            reader = csv.DictReader(f)
            for row in reader:
                performance_data.append({
                    "model": row["Model"],
                    "accuracy": float(row["Accuracy"]),
                    "precision": float(row["Precision"]),
                    "recall": float(row["Recall"]),
                    "f1_score": float(row["F1-Score"])
                })
        return performance_data
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

if __name__ == "__main__":
    uvicorn.run("web_app:app", host="127.0.0.1", port=8000, reload=True)
