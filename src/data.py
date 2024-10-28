import pandas as pd


def make_dataset(filename):
    df = pd.read_csv(filename)
    if "video_name" not in df.columns or "is_comic" not in df.columns:
        raise ValueError(
            "Le fichier CSV doit contenir les colonnes 'video_name' et 'is_comic'"
        )
    return df