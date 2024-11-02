import pickle

import numpy as np
from sklearn.compose import ColumnTransformer
from sklearn.datasets import fetch_openml
from sklearn.feature_selection import chi2
from sklearn.feature_selection import SelectPercentile
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report
from sklearn.metrics import confusion_matrix
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder
from sklearn.preprocessing import StandardScaler

np.random.seed(0)


def main():
    # load data
    X, y = fetch_openml("titanic", version=1, as_frame=True, return_X_y=True)

    # split train test data
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=0
    )

    # column names to use
    subset_feature = ["embarked", "sex", "pclass", "age", "fare"]
    X_train, X_test = X_train[subset_feature], X_test[subset_feature]

    # preprocess numeric features
    numeric_features = ["age", "fare"]
    numeric_transformer = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="median")),
            ("scaler", StandardScaler()),
        ]
    )

    # preprocess categorical features
    categorical_features = ["embarked", "sex", "pclass"]
    categorical_transformer = Pipeline(
        steps=[
            ("encoder", OneHotEncoder(handle_unknown="ignore")),
            ("selector", SelectPercentile(chi2, percentile=50)),
        ]
    )

    # combine preprocess numeric and categorical features
    preprocessor = ColumnTransformer(
        [
            ("num", numeric_transformer, numeric_features),
            ("cat", categorical_transformer, categorical_features),
        ]
    )

    # preprocess data
    X_train_processed = preprocessor.fit_transform(X_train, y_train)
    X_test_processed = preprocessor.transform(X_test)

    # train model
    clf = LogisticRegression()
    clf.fit(X_train_processed, y_train)

    # make predictions on the testing data
    y_predict = clf.predict(X_test_processed)

    # check results
    print(confusion_matrix(y_test, y_predict))
    print(classification_report(y_test, y_predict))

    # save the iris classification model as a pickle file
    prepro_pkl_file = "model/titanic_preprocessor.pkl"
    model_pkl_file = "model/titanic_classifier_model.pkl"

    with open(prepro_pkl_file, "wb") as file:
        pickle.dump(preprocessor, file)
    with open(model_pkl_file, "wb") as file:
        pickle.dump(clf, file)


if __name__ == "__main__":
    main()
