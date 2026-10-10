# Darija Moderation API — Integration Docs

## Endpoint

```
POST https://<your-modal-app-name>--fastapi-app.modal.run/moderate
Content-Type: application/json
```

## Request

```json
{
  "text": "your message here"
}
```

| Field | Type   | Required | Description         |
|-------|--------|----------|---------------------|
| text  | string | Yes      | Message to classify |

## Response

```json
{
  "flagged": true,
  "confidence": 0.9721,
  "label": "toxic"
}
```

| Field      | Type    | Description                                    |
|------------|---------|------------------------------------------------|
| flagged    | bool    | `true` if toxic with confidence > 0.85         |
| confidence | float   | Model confidence score (0.0 – 1.0)             |
| label      | string  | `"toxic"` or `"safe"`                          |

## Health Check

```
GET https://<your-modal-app-name>--fastapi-app.modal.run/health
```

Returns `{"status": "healthy"}`.

## Discord Bot Example (Python)

```python
import requests

API_URL = "https://<your-modal-app-name>--fastapi-app.modal.run/moderate"

async def is_toxic(text: str) -> bool:
    resp = requests.post(API_URL, json={"text": text}, timeout=10)
    resp.raise_for_status()
    return resp.json()["flagged"]
```

## Error Codes

| Status | Meaning                   |
|--------|---------------------------|
| 400    | Empty text                |
| 422    | Invalid request format    |
| 500    | Server error              |

## Notes

- Messages under 128 tokens are truncated automatically.
- The model handles both Arabic script and Arabizi (Latin chat script).
- Threshold is 0.85 confidence — adjust in `modal_app.py` if needed.
