import pandas as pd
import os
from scipy.io import arff

def arff_to_csv(arff_file_path, csv_file_path=None):
    """
    Convert an ARFF file to a CSV file.
    Parameters:
        arff_file_path (str): The input path of the ARFF file.
        csv_file_path (str): The output path of the CSV file. If not provided, same name as ARFF with .csv extension.
    """
    # Load the ARFF file
    data, meta = arff.loadarff(arff_file_path)
    
    # Convert the ARFF data to a DataFrame
    df = pd.DataFrame(data)
    
    # Decode byte strings in object columns (nominal/string attributes)
    for col in df.columns:
        if pd.api.types.is_object_dtype(df[col]):
            df[col] = df[col].str.decode('utf-8')
    
    # Determine CSV file name if not provided
    if csv_file_path is None:
        csv_file_path = os.path.splitext(arff_file_path)[0] + ".csv"
    
    # Save as CSV
    df.to_csv(csv_file_path, index=False)
    print(f"File saved as {csv_file_path}")

# Example usage:
arff_file_path = "WearableComputing.arff"
arff_to_csv(arff_file_path)
