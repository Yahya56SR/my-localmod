import modal

MODEL_REPO = "https://github.com/Yahya56SR/my-localmod.git"
MODEL_DIR = "/model"

model_image = (
    modal.Image.debian_slim(python_version="3.10")
    .pip_install(
        "torch>=2.2.0",
        "transformers>=4.38.2",
    )
    .apt_install("git", "git-lfs")
    .run_commands(
        f"git clone {MODEL_REPO} /tmp/repo",
        "cd /tmp/repo && git lfs pull",
        f"mv /tmp/repo/darija_moderator_model {MODEL_DIR}",
        "rm -rf /tmp/repo",
    )
)

web_image = modal.Image.debian_slim(python_version="3.10").pip_install(
    "fastapi>=0.110.0",
    "uvicorn>=0.28.0",
    "pydantic>=2.6.4",
)

app = modal.App("darija-moderation-api")


@app.cls(image=model_image, scaledown_window=300)
class ModerationModel:
    @modal.enter()
    def load(self):
        import torch
        from transformers import AutoTokenizer, AutoModelForSequenceClassification

        self.tokenizer = AutoTokenizer.from_pretrained(MODEL_DIR)
        self.model = AutoModelForSequenceClassification.from_pretrained(MODEL_DIR)
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.model.to(self.device)
        self.model.eval()

    @modal.method()
    def moderate(self, text: str):
        import torch

        inputs = self.tokenizer(text, return_tensors="pt", truncation=True, max_length=128)
        inputs = {k: v.to(self.device) for k, v in inputs.items()}

        with torch.no_grad():
            probs = torch.softmax(self.model(**inputs).logits, dim=1)

        confidence, pred_idx = probs.max(dim=1)
        label = pred_idx.item()
        score = confidence.item()

        return {
            "flagged": bool(label == 1 and score > 0.85),
            "confidence": round(score, 4),
            "label": "toxic" if label == 1 else "safe",
        }


@app.function(image=web_image)
@modal.asgi_app()
def web():
    from fastapi import FastAPI, HTTPException
    from pydantic import BaseModel

    api = FastAPI(title="Darija Content Moderation API", version="1.0")
    model = ModerationModel()

    class ModerationRequest(BaseModel):
        text: str

    class ModerationResponse(BaseModel):
        flagged: bool
        confidence: float
        label: str

    @api.post("/moderate", response_model=ModerationResponse)
    def moderate_text(request: ModerationRequest):
        if not request.text.strip():
            raise HTTPException(status_code=400, detail="Text cannot be empty.")
        return ModerationResponse(**model.moderate.remote(request.text))

    @api.get("/health")
    def health_check():
        return {"status": "healthy"}

    return api