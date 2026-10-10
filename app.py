import torch
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from transformers import AutoTokenizer, AutoModelForSequenceClassification

app = FastAPI(title="Darija Content Moderation API", version="1.0")

MODEL_PATH = "./darija_moderator_model"

print("Loading model...")
tokenizer = AutoTokenizer.from_pretrained(MODEL_PATH)
model = AutoModelForSequenceClassification.from_pretrained(MODEL_PATH)
model.eval()

device = "cuda" if torch.cuda.is_available() else "cpu"
model.to(device)


class ModerationRequest(BaseModel):
    text: str


class ModerationResponse(BaseModel):
    flagged: bool
    confidence: float
    label: str


@app.post("/moderate", response_model=ModerationResponse)
def moderate_text(request: ModerationRequest):
    if not request.text.strip():
        raise HTTPException(status_code=400, detail="Text cannot be empty.")

    inputs = tokenizer(request.text, return_tensors="pt", truncation=True, max_length=128)
    inputs = {k: v.to(device) for k, v in inputs.items()}

    with torch.no_grad():
        outputs = model(**inputs)
        probs = torch.softmax(outputs.logits, dim=1)
        confidence, pred_idx = torch.max(probs, dim=1)

    pred_label = pred_idx.item()
    conf_score = confidence.item()

    is_toxic = bool(pred_label == 1 and conf_score > 0.85)

    return ModerationResponse(
        flagged=is_toxic,
        confidence=round(conf_score, 4),
        label="toxic" if pred_label == 1 else "safe"
    )


@app.get("/health")
def health_check():
    return {"status": "healthy"}
