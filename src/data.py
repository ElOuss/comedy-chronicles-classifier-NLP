import pandas as pd


def make_dataset(filename, is_training=True):
    df = pd.read_csv(filename)
    required_columns = ["video_name"]
    
    # For training, check both columns
    if is_training:
        required_columns.append("is_comic")
    
    # Check if required columns are present
    for col in required_columns:
        if col not in df.columns:
            raise ValueError(
                f"Le fichier CSV doit contenir les colonnes {', '.join(required_columns)}"
            )
    return df