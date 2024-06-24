import pandas as pd
from sklearn.preprocessing import StandardScaler

class DataProcessor:
    def __init__(self, df: pd.DataFrame):
        """
        Initialize the DataProcessor class with a DataFrame.

        :param df: pd.DataFrame, the DataFrame to process.
        """
        self.df = df

    def remove_columns(self, columns: list):
        """
        Remove specified columns from the DataFrame.

        :param columns: list, columns to be removed.
        :return: self, to allow for method chaining.
        """
        existing_columns = [col for col in columns if col in self.df.columns]
        missing_columns = [col for col in columns if col not in self.df.columns]
        if missing_columns:
            print(f"Warning: Columns not found and cannot be removed: {missing_columns}")
        self.df.drop(existing_columns, axis=1, inplace=True)
        return self
    
    def standardize_numerical(self, columns: list):
        """
        Standardize the numerical columns in the DataFrame using StandardScaler.

        :param columns: list, numerical columns to standardize.
        :return: self, to allow for method chaining.
        """
        numerical_columns = self.df.select_dtypes(['int64', 'float64']).columns
        valid_columns = [col for col in columns if col in numerical_columns]
        invalid_columns = [col for col in columns if col not in numerical_columns]
        if invalid_columns:
            print(f"Warning: Non-numerical columns provided: {invalid_columns}")
        scaler = StandardScaler()
        self.df[valid_columns] = scaler.fit_transform(self.df[valid_columns])
        return self
    
    def get_data(self):
        """
        Return the processed DataFrame.

        :return: pd.DataFrame, the processed DataFrame.
        """
        return self.df

# Usage example:
if __name__ == "__main__":
    # Load your data
    data = pd.read_csv('../data/Churn_Modelling.csv')

    # Create the DataProcessor instance
    processor = DataProcessor(data)

    # Process the data by removing columns and standardizing numerical columns
    processed_data = (
        processor
        .remove_columns(['RowNumber', 'CustomerId', 'Surname', 'Geography', 'Gender'])
        .standardize_numerical(['CreditScore', 'Age', 'Balance', 'NumOfProducts', 'EstimatedSalary'])
        .get_data()
    )

    # Preview the processed data
    print(processed_data.head())
