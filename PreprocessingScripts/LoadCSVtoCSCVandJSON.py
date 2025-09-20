import pandas as pd
import json
import os

def load_csv_with_headers(file_path, delimiter=','):
    """
    Load data from a CSV file into a pandas DataFrame.
    Initially load with original headers so user can decide how many rows to drop.
    """
    df = pd.read_csv(file_path, delimiter=delimiter)
    return df

if __name__ == "__main__":
    # Example usage
    name = "WearableComputing_hotencoded"     # base name for files0
    file_path = name+".csv"
    df = load_csv_with_headers(file_path)

    print("DataFrame loaded with original headers:")
    print(df.head(10))  # Show first 10 rows to help user decide how many rows to drop
    print(df.info())

    # Prompt user for how many rows to drop from the top
    rows_to_drop = input("Enter how many rows to drop from the top before the actual data starts: ").strip()
    if rows_to_drop.isdigit():
        rows_to_drop = int(rows_to_drop)
    else:
        raise ValueError("Number of rows to drop must be a valid integer.")

    if rows_to_drop < 0 or rows_to_drop >= df.shape[0]:
        raise IndexError("Number of rows to drop is out of range.")

    # Drop the specified number of rows from the top
    df = df.iloc[rows_to_drop:].reset_index(drop=True)

    # Remove column labels and replace with numeric indices
    df.columns = range(df.shape[1])

    print("DataFrame after dropping rows and resetting column names:")
    print(df.head())
    print(df.info())

    # Create a directory to store output
    output_dir = os.path.join("datasets", name)
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    # Save DataFrame as CSV without headers
    csv_file = os.path.join(output_dir, name + ".csv")
    df.to_csv(csv_file, index=False, header=False)

    # Prompt user for target column index
    target_col_idx = input("Enter the index (0-based) of the target column: ").strip()
    if target_col_idx.isdigit():
        target_col_idx = int(target_col_idx)
    else:
        raise ValueError("Target column index must be a valid integer.")

    if target_col_idx < 0 or target_col_idx >= df.shape[1]:
        raise IndexError("Target column index is out of range.")

    num_samples = df.shape[0]
    num_features = df.shape[1] - 1
    feature_indices = [i for i in range(df.shape[1]) if i != target_col_idx]

    metadata = {
        "target_column_index": target_col_idx,
        "num_features": num_features,
        "num_samples": num_samples,
        "feature_indices": feature_indices
    }

    metadata_file = os.path.join(output_dir, name + "_metadata.json")
    with open(metadata_file, "w") as f:
        json.dump(metadata, f, indent=4)

    print(f"\nCSV saved to {csv_file}")
    print(f"Metadata saved to {metadata_file}")