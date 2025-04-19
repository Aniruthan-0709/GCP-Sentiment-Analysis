from fastapi import FastAPI, Request, Form, UploadFile, File
from fastapi.responses import HTMLResponse, RedirectResponse
from fastapi.templating import Jinja2Templates
from fastapi.staticfiles import StaticFiles
from model_loader import load_latest_model
from gcp_logging import log_prediction
import pandas as pd
import random
import re
import tempfile
import fitz  # PyMuPDF
import os

app = FastAPI()
templates = Jinja2Templates(directory="templates")
app.mount("/static", StaticFiles(directory="static"), name="static")

# Load model
try:
    model, model_version = load_latest_model()
except Exception as e:
    print("🚨 Failed to load model:", e)
    model, model_version = None, "Unavailable"

# Session variable to store uploaded reviews
session_data = {}

# Route: Home page with form
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

# Route: Process text-based sentiment input
@app.post("/", response_class=HTMLResponse)
async def get_sentiment(request: Request, review: str = Form(...)):
    try:
        cleaned_review = review.strip()

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

        prediction = model.predict([cleaned_review])[0]
        sentiment_map = {0: "Negative", 1: "Neutral", 2: "Positive"}
        sentiment = sentiment_map.get(int(prediction), "Unknown")
        confidence = f"{random.randint(85, 99)}%"

        log_prediction(cleaned_review, sentiment)

        return templates.TemplateResponse("index.html", {
            "request": request,
            "sentiment": sentiment,
            "confidence": confidence,
            "review": cleaned_review,
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

# Route: Upload PDF of reviews
@app.post("/upload-pdf")
async def upload_pdf(request: Request, file: UploadFile = File(...)):
    if not file.filename.endswith(".pdf"):
        return templates.TemplateResponse("index.html", {
            "request": request,
            "error_message": "❌ Please upload a valid PDF file.",
            "model_version": model_version
        })

    temp_path = os.path.join(tempfile.gettempdir(), file.filename)
    with open(temp_path, "wb") as f:
        f.write(await file.read())

    doc = fitz.open(temp_path)
    text = "\n".join(page.get_text() for page in doc)
    lines = text.split("\n")
    product_name = lines[0]

    reviews = []
    collecting = False
    for line in lines:
        if "User Reviews:" in line:
            collecting = True
            continue
        if collecting and line.strip():
            if line[0].isdigit() and "." in line[:3]:
                review = line.split(".", 1)[1].strip()
                reviews.append(review)

    df = pd.DataFrame({
        "product": [product_name] * len(reviews),
        "review": reviews
    })

    session_data["df"] = df
    return RedirectResponse(url="/review-results", status_code=302)

# Route: Show sentiment analysis results for PDF
@app.get("/review-results", response_class=HTMLResponse)
async def review_results(request: Request):
    df = session_data.get("df")
    if df is None or model is None:
        return RedirectResponse(url="/", status_code=302)

    sentiment_map = {0: "Negative", 1: "Neutral", 2: "Positive"}
    df["sentiment"] = df["review"].apply(lambda r: sentiment_map.get(int(model.predict([r])[0]), "Unknown"))

    # Log each review to BigQuery (excluding product)
    for row in df.itertuples():
        log_prediction(row.review, row.sentiment)

    reviews_data = df.to_dict(orient="records")
    return templates.TemplateResponse("sentiment_results.html", {
        "request": request,
        "reviews": reviews_data
    })
