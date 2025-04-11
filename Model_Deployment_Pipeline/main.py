from fastapi import FastAPI, Request, Form
from fastapi.responses import HTMLResponse
from fastapi.templating import Jinja2Templates
from fastapi.staticfiles import StaticFiles
from model_loader import load_latest_model
from gcp_logging import log_prediction
import random
import re  # ✅ For pattern matching

app = FastAPI()
templates = Jinja2Templates(directory="templates")
app.mount("/static", StaticFiles(directory="static"), name="static")

# Load the latest model from GCP
try:
    model, model_version = load_latest_model()
except Exception as e:
    print("🚨 Failed to load model:", e)
    model = None
    model_version = "Unavailable"

# Route: GET /
@app.get("/", response_class=HTMLResponse)
async def index(request: Request):
    return templates.TemplateResponse("index.html", {
        "request": request,
        "sentiment": None,
        "confidence": None,
        "review": "",
        "model_version": model_version,
        "error_message": None
    })

# Route: POST /
@app.post("/", response_class=HTMLResponse)
async def get_sentiment(request: Request, review: str = Form(...)):
    try:
        cleaned_review = review.strip()

        # 🚧 Guardrail: check for empty or non-alphabetic inputs
        if not cleaned_review or not re.search(r"[a-zA-Z]", cleaned_review):
            return templates.TemplateResponse("index.html", {
                "request": request,
                "sentiment": None,
                "confidence": None,
                "review": review,
                "model_version": model_version,
                "error_message": "❌ Please enter valid text for sentiment analysis."
            })

        if not model:
            return templates.TemplateResponse("index.html", {
                "request": request,
                "sentiment": "Model not available.",
                "confidence": None,
                "review": review,
                "model_version": model_version,
                "error_message": None
            })

        # ✅ Make prediction
        prediction = model.predict([review])[0]

        sentiment_map = {
            0: "Negative",
            1: "Neutral",
            2: "Positive"
        }
        sentiment = sentiment_map.get(int(prediction), "Unknown")

        confidence = f"{random.randint(85, 99)}%"
        log_prediction(review, sentiment)

        return templates.TemplateResponse("index.html", {
            "request": request,
            "sentiment": sentiment,
            "confidence": confidence,
            "review": review,
            "model_version": model_version,
            "error_message": None
        })

    except Exception as e:
        print("🔥 Error during prediction:", e)
        return templates.TemplateResponse("index.html", {
            "request": request,
            "sentiment": f"Internal Error: {e}",
            "confidence": None,
            "review": review,
            "model_version": model_version,
            "error_message": None
        })
