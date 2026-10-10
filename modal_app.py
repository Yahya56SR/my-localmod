import modal

app = modal.App("darija-moderation-api")

image = (
    modal.Image.debian_slim(python_version="3.10")
    .pip_install(
        "fastapi>=0.110.0",
        "uvicorn>=0.28.0",
        "torch>=2.2.0",
        "transformers>=4.38.2",
        "pydantic>=2.6.4",
    )
)


@app.cls(image=image, scaledown_window=300)
class ModerationModel:
    @modal.enter()
    def load_model(self):
        import torch
        from transformers import AutoTokenizer, AutoModelForSequenceClassification

        model_name = "UBC-NLP/MARBERTv2"
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.model = AutoModelForSequenceClassification.from_pretrained(model_name, num_labels=2)
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.model.to(self.device)
        self.model.eval()

    @modal.method()
    def moderate(self, text: str):
        import torch

        inputs = self.tokenizer(text, return_tensors="pt", truncation=True, max_length=128)
        inputs = {k: v.to(self.device) for k, v in inputs.items()}

        with torch.no_grad():
            outputs = self.model(**inputs)
            probs = torch.softmax(outputs.logits, dim=1)
            confidence, pred_idx = torch.max(probs, dim=1)

        pred_label = pred_idx.item()
        conf_score = confidence.item()
        is_toxic = bool(pred_label == 1 and conf_score > 0.85)

        return {
            "flagged": is_toxic,
            "confidence": round(conf_score, 4),
            "label": "toxic" if pred_label == 1 else "safe"
        }


@app.local_entrypoint()
def test():
    model = ModerationModel()
    result = model.moderate.remote("this is a test")
    print(result)
