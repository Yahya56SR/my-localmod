import modal
from modal_app import app

@app.function(image=modal.Image.debian_slim(python_version="3.10").pip_install("fastapi", "uvicorn"))
@modal.asgi_app()
def fastapi_app():
    from fastapi import FastAPI, HTTPException
    from pydantic import BaseModel

    web = FastAPI(title="Darija Content Moderation API", version="1.0")

    model_cls = modal.Cls.from_name("darija-moderation-api", "ModerationModel")
    model = model_cls()

    class ModerationRequest(BaseModel):
        text: str

    class ModerationResponse(BaseModel):
        flagged: bool
        confidence: float
        label: str

    @web.post("/moderate", response_model=ModerationResponse)
    def moderate_text(request: ModerationRequest):
        if not request.text.strip():
            raise HTTPException(status_code=400, detail="Text cannot be empty.")

        result = model.moderate.remote(request.text)
        return ModerationResponse(**result)

    @web.get("/health")
    def health_check():
        return {"status": "healthy"}

    return web
