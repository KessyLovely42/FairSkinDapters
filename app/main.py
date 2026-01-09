import traceback
import uuid
from pathlib import Path

from fastapi import FastAPI, File, HTTPException, UploadFile, Request
from fastapi.responses import HTMLResponse, JSONResponse
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates

from core.inference import preprocess_predict_image  

from config import (
	BASE_DIR,
	TEMPLATES_DIR,
	STATIC_DIR,
	TEST_IMAGEDIR,
	CAM_IMAGEDIR,
)

app = FastAPI(title="FairSkinDapter: A fairness aware skin cancer detection API", version="1.0")

# Ensure directories exist 
TEMPLATES_DIR.mkdir(parents=True, exist_ok=True)
STATIC_DIR.mkdir(parents=True, exist_ok=True)
TEST_IMAGEDIR.mkdir(parents=True, exist_ok=True)
CAM_IMAGEDIR.mkdir(parents=True, exist_ok=True)

# Templates (HTML)
templates = Jinja2Templates(directory=str(TEMPLATES_DIR))

# Static files (CSS/JS)
app.mount("/static", StaticFiles(directory=str(STATIC_DIR)), name="static")

# Static files (CAM images)
app.mount("/cam_images", StaticFiles(directory=str(CAM_IMAGEDIR)), name="cam_images")


#Endpoint for the homepage
@app.get("/", response_class=HTMLResponse)
async def index(request: Request):
    return templates.TemplateResponse("index.html", {"request": request})

#Endpoint for predictions
@app.post("/predictions")
async def predict_image(request: UploadFile = File(...)):
    """
    Accepts an uploaded image, runs inference + CAM generation,
    returns JSON: cam_image URL + prediction + probability + confidence bucket.
    """
    try:
        # Save uploaded image
        filename = f"{uuid.uuid4()}.jpg"
        image_path = TEST_IMAGEDIR / filename
        image_path.write_bytes(await request.read())

        # Run model inference (predict() saves CAM and returns its path)
        cam_path, prediction, predict_proba = preprocess_predict_image(image_path)

        # Convert CAM path into URL served by StaticFiles
        cam_url = f"/cam_images/{Path(cam_path).name}"

        # Probability can be 0..1 or 0..100; convert to percent for confidence buckets
        p = float(predict_proba)
        pct = p * 100.0 if p <= 1.0 else p

        if pct >= 80:
            confidence = "High"
        elif pct >= 70:
            confidence = "Medium"
        elif pct >= 50:
            confidence = "Low"
        else:
            confidence = "Very Low"

        return JSONResponse(
            {
                "prediction": int(prediction),
                "label": "Malignant" if int(prediction) == 1 else "Benign",
                "predict_proba": float(predict_proba),
                "confidence_level": confidence,
                "cam_image": cam_url,
            }
        )

    except Exception as e:
        print(traceback.format_exc())
        raise HTTPException(status_code=500, detail=f"Something went wrong: {str(e)}")
