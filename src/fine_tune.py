import pandas as pd
from transformers import DistilBertTokenizer, DistilBertForSequenceClassification, Trainer, TrainingArguments
from sklearn.model_selection import train_test_split


def load_data(filename):
    """
    Charger les données et les diviser en train/test.
    """
    df = pd.read_csv(filename)
    return train_test_split(df["video_name"], df["is_comic"], test_size=0.2, random_state=42)


def prepare_model():
    """
    Charger le tokenizer et le modèle DistilBERT.
    """
    tokenizer = DistilBertTokenizer.from_pretrained("distilbert-base-uncased")
    model = DistilBertForSequenceClassification.from_pretrained("distilbert-base-uncased", num_labels=2)
    return tokenizer, model


def fine_tune_model(train_texts, train_labels, val_texts, val_labels, tokenizer, model):
    """
    Fine-tuner le modèle DistilBERT.
    """
    # Préparation des données pour le modèle
    train_encodings = tokenizer(list(train_texts), truncation=True, padding=True, max_length=128, return_tensors="pt")
    val_encodings = tokenizer(list(val_texts), truncation=True, padding=True, max_length=128, return_tensors="pt")

    # Configuration de l'entraînement
    training_args = TrainingArguments(
        output_dir="../models/distilbert_model",  # Où sauvegarder le modèle
        evaluation_strategy="epoch",
        save_strategy="epoch",
        num_train_epochs=3,
        per_device_train_batch_size=16,
        logging_dir="../logs",  # Dossier pour les logs
        logging_steps=10,
    )

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_encodings,
        eval_dataset=val_encodings,
    )

    trainer.train()


if __name__ == "__main__":
    # Charger les données
    train_texts, val_texts, train_labels, val_labels = load_data("../data/raw/train.csv")
    # Préparer le modèle
    tokenizer, model = prepare_model()
    # Fine-tuner le modèle
    fine_tune_model(train_texts, train_labels, val_texts, val_labels, tokenizer, model)
