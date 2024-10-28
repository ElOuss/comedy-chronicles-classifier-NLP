from sklearn.ensemble import RandomForestClassifier
from sklearn.naive_bayes import MultinomialNB
from sklearn.tree import DecisionTreeClassifier

def make_model(model_type="random_forest"):
    if model_type == "random_forest":
        return RandomForestClassifier()
    elif model_type == "naive_bayes":
        return MultinomialNB()
    elif model_type == "decision_tree":
        return DecisionTreeClassifier()
    else:
        raise ValueError(f"Unknown model type: {model_type}")
