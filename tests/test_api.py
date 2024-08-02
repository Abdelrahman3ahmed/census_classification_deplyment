from fastapi.testclient import TestClient
from src.api import app

client = TestClient(app)

def test_read_root():
    response = client.get("/")
    assert response.status_code == 200
    assert response.json() == {"message": "Welcome to the Census Classification API"}

def test_predict_income():
    response = client.post("/predict/", json={
        "age": 25,
        "workclass": "Private",
        "fnlwgt": 226802,
        "education": "11th",
        "education_num": 7,
        "marital_status": "Never-married",
        "occupation": "Machine-op-inspct",
        "relationship": "Own-child",
        "race": "Black",
        "sex": "Male",
        "capital_gain": 0,
        "capital_loss": 0,
        "hours_per_week": 40,
        "native_country": "United-States"
    })
    assert response.status_code == 200
    assert "prediction" in response.json()

def test_predict_income_invalid():
    response = client.post("/predict/", json={})
    assert response.status_code == 422

