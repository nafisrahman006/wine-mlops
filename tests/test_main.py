VALID_WINE = {
    "fixed_acidity": 7.4,
    "volatile_acidity": 0.3,
    "citric_acid": 0.3,
    "residual_sugar": 2.0,
    "alcohol": 12.0,
}


def test_health_check(client):
    resp = client.get("/")
    assert resp.status_code == 200
    body = resp.json()
    assert body["message"] == "Wine Quality Prediction API is live"
    assert "features" in body


def test_predict_valid_input(client):
    resp = client.post("/predict", json=VALID_WINE)
    assert resp.status_code == 200
    body = resp.json()
    assert "good_quality" in body
    assert isinstance(body["good_quality"], bool)
    assert "prediction_time_ms" in body


def test_predict_high_alcohol_is_good_quality(client):
    payload = {**VALID_WINE, "alcohol": 13.0}
    resp = client.post("/predict", json=payload)
    assert resp.status_code == 200
    assert resp.json()["good_quality"] is True


def test_predict_low_alcohol_is_not_good_quality(client):
    payload = {**VALID_WINE, "alcohol": 8.5}
    resp = client.post("/predict", json=payload)
    assert resp.status_code == 200
    assert resp.json()["good_quality"] is False


def test_predict_rejects_out_of_range_alcohol(client):
    # WineInput bounds: alcohol must be between 8 and 15
    payload = {**VALID_WINE, "alcohol": 25.0}
    resp = client.post("/predict", json=payload)
    assert resp.status_code == 422


def test_predict_rejects_negative_fixed_acidity(client):
    payload = {**VALID_WINE, "fixed_acidity": -1.0}
    resp = client.post("/predict", json=payload)
    assert resp.status_code == 422


def test_predict_rejects_missing_field(client):
    payload = {k: v for k, v in VALID_WINE.items() if k != "alcohol"}
    resp = client.post("/predict", json=payload)
    assert resp.status_code == 422


def test_predict_batch(client):
    payload = {"wines": [VALID_WINE, {**VALID_WINE, "alcohol": 9.0}]}
    resp = client.post("/predict/batch", json=payload)
    assert resp.status_code == 200
    body = resp.json()
    assert body["total"] == 2
    assert len(body["predictions"]) == 2


def test_metrics_endpoint_after_predictions(client):
    client.post("/predict", json=VALID_WINE)
    resp = client.get("/metrics")
    assert resp.status_code == 200
    body = resp.json()
    assert body["total_predictions"] >= 1
    assert "drift_status" in body
