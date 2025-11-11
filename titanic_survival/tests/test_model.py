import pandas as pd
from src.model import train


def test_train_returns_model_and_accuracy():
    # create a tiny synthetic dataset
    df = pd.DataFrame({
        'Pclass': [1, 3, 2, 3],
        'Age': [22, 38, 26, 35],
        'Fare': [7.25, 71.2833, 7.925, 8.05],
        'Sex': ['male', 'female', 'female', 'male'],
        'Survived': [0, 1, 1, 0]
    })
    model, acc, X_test, y_test = train(df, test_size=0.5)
    assert hasattr(model, 'predict')
    assert 0.0 <= acc <= 1.0
