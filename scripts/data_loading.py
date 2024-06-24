import pandas as pd
import os
import logging

logging.basicConfig(level=logging.ERROR)

class DataLoader:
    def __init__(self, data_file: str):
        """
        Initialize the DataLoader class with the path to the data file.

        :param data_file: str, path to the data file.
        """
        self.data_file = data_file
        self.df = None

    def validate_file_path(self):
        """Validate if the file path exists."""
        if not os.path.exists(self.data_file):
            logging.error(f"File does not exist: {self.data_file}")
            return False
        return True
    
    def load_data(self, file_format: str = 'csv'):
        """
        Load data from the specified file into a DataFrame.

        :param file_format: str, the format of the data file ('csv', 'excel').
        :return: DataFrame or None if an error occurs.
        """
        if not self.validate_file_path():
            return None

        try:
            if file_format == 'csv':
                self.df = pd.read_csv(self.data_file)
            elif file_format == 'excel':
                self.df = pd.read_excel(self.data_file)
            else:
                raise ValueError(f"Unsupported file format: {file_format}")
            return self.df
        except FileNotFoundError:
            logging.error(f"Error: The file {self.data_file} was not found.")
            return None
        except pd.errors.EmptyDataError:
            logging.error(f"Error: The file {self.data_file} is empty.")
            return None
        except pd.errors.ParserError:
            logging.error(f"Error: There was an error parsing the file {self.data_file}.")
            return None
        except ValueError as ve:
            logging.error(ve)
            return None
        except Exception as e:
            logging.error(f"An unexpected error occurred: {e}")
            return None

    def preview_data(self, num_rows: int = 5):
        """
        Print the first few rows of the data.

        :param num_rows: int, number of rows to preview.
        """
        if self.df is not None:
            print(self.df.head(num_rows))
        else:
            print("No data loaded to preview.")
    
    def get_data(self):
        """Return the loaded DataFrame."""
        return self.df

def main():
    loader = DataLoader("../data/Churn_Modelling.csv")

    data = loader.load_data(file_format='csv')

    if data is not None:
        loader.preview_data()

if __name__ == "__main__":
    main()
