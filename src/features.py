def make_features(df):
    y = df["is_comic"]

    # Fake X: constant equal to 0
    X = [[0]] * len(y)

    return X, y