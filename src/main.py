import click
import joblib
from sklearn.model_selection import cross_val_score
from sklearn.feature_extraction.text import CountVectorizer
import pandas as pd
from sklearn.model_selection import GridSearchCV


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
    # Load the dataset
    df = make_dataset(input_filename, is_training=True)
    
    # Get features, labels, and the vectorizer
    X, y, vectorizer = make_features(df)  # Ensure make_features returns vectorizer

    # Train the model
    model = make_model(model_type)
    model.fit(X, y)

    # Save the model and vectorizer together
    joblib.dump({"model": model, "vectorizer": vectorizer}, model_dump_filename)
    print(f"Model and vectorizer saved to {model_dump_filename}")

@click.command()
@click.option("--input_filename", default="data/raw/test.csv", help="Path to the test data CSV")
@click.option("--model_dump_filename", default="models/model.pkl", help="Path to the trained model file")
@click.option("--output_filename", default="data/processed/predictions.csv", help="Path to save predictions")
def predict(input_filename, model_dump_filename, output_filename):
    # Load the saved model and vectorizer
    saved_objects = joblib.load(model_dump_filename)
    model = saved_objects["model"]
    vectorizer = saved_objects["vectorizer"]

    # Load and transform test data
    df = make_dataset(input_filename, is_training=False)
    X_test = vectorizer.transform(df["video_name"])

    # Make predictions and save the results
    predictions = model.predict(X_test)
    prediction_df = pd.DataFrame({"video_name": df["video_name"], "is_comic": predictions})
    prediction_df.to_csv(output_filename, index=False)
    print(f"Predictions saved to {output_filename}")

@click.command()
@click.option("--input_filename", default="data/raw/train.csv", help="Path to the training data CSV")
@click.option("--model_type", default="random_forest", help="Type of model to evaluate (random_forest, naive_bayes, decision_tree)")
def evaluate(input_filename, model_type):
    # Load the dataset and extract features
    df = make_dataset(input_filename, is_training=True)
    X, y, _ = make_features(df)  # Ignore the vectorizer here as it's not needed

    # Define the parameter grid for Random Forest
    param_grid = {
        'n_estimators': [50, 100, 200],
        'max_depth': [None, 10, 20, 30],
        'min_samples_split': [2, 5, 10],
        'min_samples_leaf': [1, 2, 4]
    }

    # Initialize the model for tuning
    rf_model = make_model(model_type)

    # Set up GridSearchCV with 5-fold cross-validation
    grid_search = GridSearchCV(estimator=rf_model, param_grid=param_grid, cv=5, scoring='accuracy', n_jobs=-1)
    grid_search.fit(X, y)

    # Output the best parameters and cross-validation score
    print("Best parameters:", grid_search.best_params_)
    print("Best cross-validation score:", grid_search.best_score_)
    print("All cross-validation scores:", grid_search.cv_results_['mean_test_score'])

cli.add_command(train)
cli.add_command(predict)
cli.add_command(evaluate)

if __name__ == "__main__":
    cli()
