from fastapi.testclient import TestClient

from app.main import app, get_service


class FakeSentimentService:
    model_name = "fake-model"

    def analyze_text(self, text: str, language=None):
        return {"label": "positive", "score": 0.99, "model": self.model_name, "language": language}

    def analyze_batch(self, texts, language=None):
        return [self.analyze_text(t, language=language) for t in texts]


def test_health_endpoint():
    app.dependency_overrides[get_service] = lambda: FakeSentimentService()
    client = TestClient(app)
    response = client.get("/health")
    assert response.status_code == 200
    assert response.json()["model"] == "fake-model"
    app.dependency_overrides = {}


def test_sentiment_single():
    app.dependency_overrides[get_service] = lambda: FakeSentimentService()
    client = TestClient(app)
    response = client.post("/sentiment", json={"text": "hello", "language": "en"})
    assert response.status_code == 200
    body = response.json()
    assert body["label"] == "positive"
    assert body["language"] == "en"
    app.dependency_overrides = {}


def test_sentiment_batch():
    app.dependency_overrides[get_service] = lambda: FakeSentimentService()
    client = TestClient(app)
    response = client.post(
        "/sentiment/batch",
        json={"texts": ["hi", "hola"], "language": "es"},
    )
    assert response.status_code == 200
    body = response.json()
    assert len(body) == 2
    assert all(item["label"] == "positive" for item in body)
    app.dependency_overrides = {}
