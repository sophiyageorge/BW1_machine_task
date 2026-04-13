from fastapi.testclient import TestClient
from app.predict import app

client = TestClient(app)

# Test home endpoint
def test_home():
    response = client.get("/")
    assert response.status_code == 200
    assert response.json() == {"message": "ML Model API Running"}


# Test prediction endpoint (valid input)
def test_predict_valid():
    payload = {
        "sepal_length": 5.1,
        "sepal_width": 3.5,
        "petal_length": 1.4,
        "petal_width": 0.2
    }

    response = client.post("/predict", json=payload)

    assert response.status_code == 200
    assert "prediction" in response.json()
    assert isinstance(response.json()["prediction"], list)


# Test prediction endpoint (invalid input)
def test_predict_invalid():
    payload = {
        "sepal_length": "invalid",  # wrong type
        "sepal_width": 3.5,
        "petal_length": 1.4,
        "petal_width": 0.2
    }

    response = client.post("/predict", json=payload)

    assert response.status_code == 422  # validation error


def test_predict_edge_case():
    payload = {
        "sepal_length": 0,
        "sepal_width": 0,
        "petal_length": 0,
        "petal_width": 0
    }

    response = client.post("/predict", json=payload)

    assert response.status_code == 200