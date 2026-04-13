import os
import joblib
import numpy as np


def load_model():
    BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    model_path = os.path.join(BASE_DIR, "app", "model.pkl")
    return joblib.load(model_path)


# ✅ Test 1: Model loads correctly
def test_model_load():
    model = load_model()
    assert model is not None


# ✅ Test 2: Prediction works
def test_model_prediction():
    model = load_model()

    sample = np.array([[5.1, 3.5, 1.4, 0.2]])
    prediction = model.predict(sample)

    assert prediction is not None
    assert len(prediction) == 1


# ✅ Test 3: Output type check
def test_prediction_type():
    model = load_model()

    sample = np.array([[6.0, 3.0, 4.8, 1.8]])
    prediction = model.predict(sample)

    assert isinstance(prediction, np.ndarray)


# ✅ Test 4: Known value test (important!)
def test_known_output():
    model = load_model()

    # Known Iris Setosa example
    sample = np.array([[5.1, 3.5, 1.4, 0.2]])
    prediction = model.predict(sample)

    # Expect class 0 (Setosa)
    assert prediction[0] == 0