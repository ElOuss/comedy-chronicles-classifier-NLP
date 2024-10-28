import click
import joblib
from sklearn.model_selection import cross_val_score
from sklearn.feature_extraction.text import CountVectorizer
import pandas as pd

from data import make_dataset
from feature import make_features
from models import make_model

@click.group()
def cli():
    pass

@click.command()
@click.option("--input_filename", default="data/raw/train.csv", help="Path to the training data CSV")
@click.option("--model_dump_filename", default="models/model.pkl", help="Path to save the trained model")
@click.option("--model_type", default="random_forest", help="Type of model to train (random_forest, naive_bayes, decision_tree)")
def train(input_filename, model_dump_filename, model_type):
    df = make_dataset(input_filename)
    X, y = make_features(df)

    model = make_model(model_type)
    model.fit(X, y)

    joblib.dump((model, X), model_dump_filename)
    print(f"Model trained and saved to {model_dump_filename}")

@click.command()
@click.option("--input_filename", default="data/raw/test.csv", help="Path to the test data CSV")
@click.option("--model_dump_filename", default="models/model.pkl", help="Path to the trained model file")
@click.option("--output_filename", default="data/processed/predictions.csv", help="Path to save predictions")
def predict(input_filename, model_dump_filename, output_filename):
    model, X_train = joblib.load(model_dump_filename)
    
    # Ensuring the new data is transformed consistently
    df = make_dataset(input_filename)
    vectorizer = CountVectorizer(vocabulary=X_train.get_feature_names_out())
    X_test = vectorizer.transform(df["video_name"])

    predictions = model.predict(X_test)
    prediction_df = pd.DataFrame({"video_name": df["video_name"], "is_comic": predictions})
    prediction_df.to_csv(output_filename, index=False)
    
    print(f"Predictions saved to {output_filename}")

@click.command()
@click.option("--input_filename", default="data/raw/train.csv", help="Path to the training data CSV")
@click.option("--model_type", default="random_forest", help="Type of model to evaluate (random_forest, naive_bayes, decision_tree)")
def evaluate(input_filename, model_type):
    df = make_dataset(input_filename)
    X, y = make_features(df)

    model = make_model(model_type)
    scores = cross_val_score(model, X, y, cv=5, scoring="accuracy")
    
    print(f"Cross-validation accuracy scores: {scores}")
    print(f"Mean accuracy: {scores.mean():.2f}")

cli.add_command(train)
cli.add_command(predict)
cli.add_command(evaluate)

if __name__ == "__main__":
    cli()
