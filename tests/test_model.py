import unittest
import joblib
import pandas as pd
from sklearn.model_selection import train_test_split
from src.model import train_model

class TestModel(unittest.TestCase):
    def setUp(self):
        self.model = joblib.load('models/model.joblib')
        self.label_encoders = joblib.load('models/label_encoders.joblib')
        self.test_data = pd.read_csv('data/clean_census.csv').sample(10)

    def test_model_accuracy(self):
        X_test = self.test_data.drop('income', axis=1)
        y_test = self.test_data['income']
        predictions = self.model.predict(X_test)
        accuracy = (predictions == y_test).mean()
        self.assertGreater(accuracy, 0.7)

    def test_model_slice_performance(self):
        slice_data = self.test_data[self.test_data['sex'] == 1]  # Example slice
        X_test = slice_data.drop('income', axis=1)
        y_test = slice_data['income']
        predictions = self.model.predict(X_test)
        accuracy = (predictions == y_test).mean()
        self.assertGreater(accuracy, 0.7)

if __name__ == "__main__":
    unittest.main()

