from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score, confusion_matrix, classification_report
from joblib import dump, load
import os

class ModelTrainer:
    def __init__(self, model_type: str):
        """
        Initialize the ModelTrainer class with the specified model type.

        :param model_type: str, type of model to initialize ('logistic_regression' or 'decision_tree').
        """
        self.model_type = model_type
        self.model = None
        self._initialize_model()

    def _initialize_model(self):
        """Initialize the model based on the specified model type."""
        if self.model_type == 'logistic_regression':
            self.model = LogisticRegression()
        elif self.model_type == 'decision_tree':
            self.model = DecisionTreeClassifier()
        else:
            raise ValueError(f"Unsupported model type: {self.model_type}")

    def train(self, X_train, y_train):
        """
        Train the model using the training data.

        :param X_train: np.array, features for training.
        :param y_train: np.array, target labels for training.
        """
        self.model.fit(X_train, y_train)

    def evaluate(self, X_test, y_test):
        """
        Evaluate the model using the test data.

        :param X_test: np.array, features for testing.
        :param y_test: np.array, target labels for testing.
        :return: dict, evaluation metrics.
        """
        y_pred = self.model.predict(X_test)
        accuracy = accuracy_score(y_test, y_pred)
        precision = precision_score(y_test, y_pred, average='binary')
        recall = recall_score(y_test, y_pred, average='binary')
        confusion_mat = confusion_matrix(y_test, y_pred)
        classification_rep = classification_report(y_test, y_pred)
        
        metrics = {
            'accuracy': accuracy,
            'precision': precision,
            'recall': recall,
            'confusion_matrix': confusion_mat,
            'classification_report': classification_rep
        }
        return metrics

    def save_model(self, file_path: str):
        """
        Save the trained model to a file.

        :param file_path: str, path to save the model.
        """
        # Ensure the directory exists
        os.makedirs(os.path.dirname(file_path), exist_ok=True)
        
        # Save the model
        dump(self.model, file_path)
        print(f"Model saved to {file_path}")

    def load_model(self, file_path: str):
        """
        Load a model from a file.

        :param file_path: str, path to load the model from.
        """
        self.model = load(file_path)
        print(f"Model loaded from {file_path}")
