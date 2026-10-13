from sklearn.linear_model import (
    LinearRegression,
    Ridge,
    Lasso
)

from sklearn.preprocessing import PolynomialFeatures
from sklearn.pipeline import Pipeline

from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor


def get_regression_models():

    return {

        "Linear Regression":
            LinearRegression(),

        "Polynomial Regression":
            Pipeline([
                (
                    "poly",
                    PolynomialFeatures(degree=2)
                ),
                (
                    "linear",
                    LinearRegression()
                )
            ]),

        "Ridge Regression":
            Ridge(alpha=1.0),

        "Lasso Regression":
            Lasso(alpha=0.01),

        "Decision Tree":
            DecisionTreeRegressor(
                random_state=42
            ),

        "Random Forest":
            RandomForestRegressor(
                n_estimators=100,
                random_state=42
            )
    }


def train_regression_model(
    model_name,
    X_train,
    X_test,
    y_train,
    y_test
):

    models = get_regression_models()

    model = models[model_name]

    model.fit(X_train, y_train)

    predictions = model.predict(X_test)

    return model, predictions