import pandas as pd
from sklearn.model_selection import train_test_split
from data_loading import DataLoader
from data_processing import DataProcessor
from model_trainer import ModelTrainer

# Define paths and parameters
data_file = "../data/Churn_Modelling.csv"
logistic_regression_model_file = "models/logistic_regression_model.joblib"
decision_tree_model_file = "models/decision_tree_model.joblib"

if __name__ == "__main__":
    # Step 1: Load and process data
    loader = DataLoader(data_file)
    data = loader.load_data(file_format='csv')

    if data is None:
        print("Error loading data. Exiting.")
        exit()

    processor = DataProcessor(data)
    processed_data = (
        processor
        .remove_columns(['RowNumber', 'CustomerId', 'Surname', 'Geography', 'Gender'])
        .standardize_numerical(['CreditScore', 'Age', 'Balance', 'NumOfProducts', 'EstimatedSalary'])
        .get_data()
    )
    
    # Step 2: Split data into train and test sets
    X = processed_data.drop('Exited', axis=1)
    y = processed_data['Exited']
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

    # Step 3: Train logistic regression model
    trainer = ModelTrainer(model_type='logistic_regression')
    trainer.train(X_train, y_train)
    
    # Evaluate logistic regression model
    evaluation_results = trainer.evaluate(X_test, y_test)
    
    # Print evaluation results including classification report
    print("Logistic Regression Evaluation:")
    print("Accuracy:", evaluation_results['accuracy'])
    print("Precision:", evaluation_results['precision'])
    print("Recall:", evaluation_results['recall'])
    print("Confusion Matrix:\n", evaluation_results['confusion_matrix'])
    print("Classification Report:\n", evaluation_results['classification_report'])

    # Save logistic regression model
    trainer.save_model(logistic_regression_model_file)

    # Step 4: Train decision tree model
    trainer_dt = ModelTrainer(model_type='decision_tree')
    trainer_dt.train(X_train, y_train)
    
    # Evaluate decision tree model
    evaluation_results_dt = trainer_dt.evaluate(X_test, y_test)
    
    # Print evaluation results including classification report
    print("Decision Tree Evaluation:")
    print("Accuracy:", evaluation_results_dt['accuracy'])
    print("Precision:", evaluation_results_dt['precision'])
    print("Recall:", evaluation_results_dt['recall'])
    print("Confusion Matrix:\n", evaluation_results_dt['confusion_matrix'])
    print("Classification Report:\n", evaluation_results_dt['classification_report'])

    # Save decision tree model
    trainer_dt.save_model(decision_tree_model_file)
