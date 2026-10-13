from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC


def get_classification_models():

    return {
        "Logistic Regression": LogisticRegression(
            max_iter=1000
        ),

        "K-Nearest Neighbors": KNeighborsClassifier(),

        "Decision Tree": DecisionTreeClassifier(
            random_state=42
        ),

        "Random Forest": RandomForestClassifier(
            n_estimators=100,
            random_state=42
        ),

        "Support Vector Machine": SVC(
            probability=True,
            random_state=42
        )
    }


def train_classification_model(
    model_name,
    X_train,
    X_test,
    y_train,
    y_test
):

    models = get_classification_models()

    model = models[model_name]

    model.fit(X_train, y_train)

    predictions = model.predict(X_test)

    return model, predictions