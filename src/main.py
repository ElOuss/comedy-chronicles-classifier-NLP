import click
import joblib
from sklearn.model_selection import cross_val_score,train_test_split, GridSearchCV
from sklearn.metrics import classification_report, accuracy_score, confusion_matrix
import pandas as pd
import logging
from data import make_dataset
from feature import make_features
from models import make_model
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np

# Set up logging configuration
logging.basicConfig(filename="metrics_log.txt", level=logging.INFO,
                    format="%(asctime)s - %(levelname)s - %(message)s")

def log_metrics(params, accuracy, report, cv_results):
    """Logs metrics for model evaluation."""
    logging.info(f"Parameters: {params}")
    logging.info(f"Accuracy: {accuracy}")
    logging.info(f"Classification Report:\n{report}")
    logging.info(f"Cross-validation results:\n{cv_results}")

@click.group()
def cli():
    pass

@click.command()
@click.option("--input_filename", default="data/raw/train.csv", help="Path to the training data CSV")
@click.option("--model_dump_filename", default="models/model.pkl", help="Path to save the trained model")
@click.option("--model_type", default="random_forest", help="Type of model to train (random_forest, naive_bayes, decision_tree)")
@click.option("--use_tfidf", is_flag=True, help="Use TF-IDF for feature extraction instead of CountVectorizer")
def train(input_filename, model_dump_filename, model_type, use_tfidf):
    # Load the dataset
    df = make_dataset(input_filename, is_training=True)
    
    # Get features, labels, and the vectorizer
    X, y, vectorizer = make_features(df, use_tfidf=use_tfidf)

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

    # Make predictions
    predictions = model.predict(X_test)
    prediction_df = pd.DataFrame({"video_name": df["video_name"], "is_comic": predictions})
    prediction_df.to_csv(output_filename, index=False)
    print(f"Predictions saved to {output_filename}")

    # Check if true labels are available to plot the confusion matrix
    if "is_comic" in df.columns:
        y_test = df["is_comic"]
        plot_confusion_matrix(y_test, predictions)
    else:
        print("True labels are not available in the test data; skipping confusion matrix.")

@click.command()
@click.option("--input_filename", default="data/raw/train.csv", help="Path to the training data CSV")
@click.option("--model_type", default="random_forest", help="Type of model to evaluate (random_forest, naive_bayes, decision_tree)")
@click.option("--use_tfidf", is_flag=True, help="Use TF-IDF for feature extraction instead of CountVectorizer")
def evaluate(input_filename, model_type, use_tfidf):
    # Load the dataset and extract features
    df = make_dataset(input_filename, is_training=True)
    X, y, _ = make_features(df, use_tfidf=use_tfidf)

    # Define the parameter grid for Random Forest
    param_grid = {
        'n_estimators': [50, 100, 200],
        'max_depth': [None, 10, 20],
        'min_samples_split': [2, 5, 10],
        'min_samples_leaf': [1, 2, 4]
    }

    # Initialize the model for tuning
    rf_model = make_model(model_type)

    # Set up GridSearchCV with 5-fold cross-validation
    grid_search = GridSearchCV(estimator=rf_model, param_grid=param_grid, cv=5, scoring='accuracy', n_jobs=-1)
    grid_search.fit(X, y)

    # Get the best model and parameters
    best_model = grid_search.best_estimator_
    best_params = grid_search.best_params_
    best_score = grid_search.best_score_
    cv_results = grid_search.cv_results_['mean_test_score']

    # Log metrics
    log_metrics(best_params, best_score, None, cv_results)
    print("Best parameters:", best_params)
    print("Best cross-validation score:", best_score)

    # Generate predictions on the training data and plot the confusion matrix
    y_pred = best_model.predict(X)
    plot_confusion_matrix(y, y_pred)  # Call the plot function here to visualize



def plot_confusion_matrix(y_true, y_pred):
    cm = confusion_matrix(y_true, y_pred)
    plt.figure(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues')
    plt.xlabel('Predicted Label')
    plt.ylabel('True Label')
    plt.title('Confusion Matrix')
    plt.show()


def plot_top_tfidf_features(vectorizer, model, top_n=10):
    feature_names = np.array(vectorizer.get_feature_names_out())
    
    # Get the importance or weights for each feature from the model
    if hasattr(model, 'coef_'):  # For models like LogisticRegression
        importance = np.abs(model.coef_).flatten()
    elif hasattr(model, 'feature_importances_'):  # For models like RandomForest
        importance = model.feature_importances_
    else:
        print("Model does not support feature importance extraction.")
        return

    # Sort and get the top features
    top_features_idx = np.argsort(importance)[-top_n:]
    top_features = feature_names[top_features_idx]
    top_importance = importance[top_features_idx]

    # Plot
    plt.figure(figsize=(10, 6))
    plt.barh(top_features, top_importance, color='skyblue')
# Call this function after training the model with TF-IDF features
# For example:
# plot_top_tfidf_features(vectorizer, best_model)

# Call this function after training the model with TF-IDF features
# For example:
# plot_top_tfidf_features(vectorizer, best_model)

cli.add_command(train)
cli.add_command(predict)
cli.add_command(evaluate)

if __name__ == "__main__":
    cli()
