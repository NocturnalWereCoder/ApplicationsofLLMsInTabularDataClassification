import pandas as pd
import os

def hot_encode_csv(file_path, sep=',', target_index=None, threshold=10, remove_index=None):
    # 1. Load the CSV with specified separator
    df = pd.read_csv(file_path, sep=sep)
    
    # 2. Drop a column by index, if desired
    if remove_index is not None:
        if 0 <= remove_index < len(df.columns):
            column_to_remove = df.columns[remove_index]
            df.drop(columns=[column_to_remove], inplace=True)
            print(f"Removed column '{column_to_remove}' at index {remove_index}.")
        else:
            print(f"Invalid index {remove_index}. No column removed.")


    # --- Force columns containing only "True" / "False" to 1 / 0 ---
    for col in df.columns:
        if df[col].dtype == 'object':
            unique_vals = set(df[col].dropna().unique())
            # Check if the column ONLY contains 'True' or 'False' (strings)
            if unique_vals.issubset({"True", "False"}):
                df[col] = df[col].map({"True": 1, "False": 0})

    # 5. One-hot encode any remaining object columns
    obj_cols = df.select_dtypes(include=["object"]).columns.tolist()
    if obj_cols:  # If there's anything left to encode
        df = pd.get_dummies(df, columns=obj_cols, drop_first=True)

    # 6. If any columns ended up as actual bool dtype, convert them to int
    bool_cols = df.select_dtypes(include="bool").columns
    if len(bool_cols) > 0:
        df[bool_cols] = df[bool_cols].astype(int)
    
    # 7. Replace contents of the target column by index with binary pass/fail
    if target_index is not None:
        if 0 <= target_index < len(df.columns):
            target_column = df.columns[target_index]
            df[target_column] = (df[target_column] >= threshold).astype(int)
            print(f"Replaced contents of column '{target_column}' at index {target_index} with binary values (threshold={threshold}).")
        else:
            print(f"Invalid target index {target_index}. No conversion applied.")

    # 8. Save the transformed CSV
    base_name = os.path.splitext(file_path)[0]
    new_file_name = f"{base_name}_hotencoded.csv"
    df.to_csv(new_file_name, index=False)
    print(f"File saved as {new_file_name}")


file_path = "WearableComputing.csv"  
remove_index = None  # Do not remove any column in this case
target_index = None
threshold = None  # Threshold for pass/fail
sep = ','  # CSV separator
hot_encode_csv(file_path, sep=sep, target_index=target_index, threshold=threshold, remove_index=remove_index)
