import os
import mlflow
import mlflow.sklearn
from sklearn.preprocessing import StandardScaler
from sklearn.datasets import fetch_california_housing
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.tree import DecisionTreeRegressor
from sklearn.metrics import mean_squared_error
import pandas as pd
import tempfile


import pandas as pd

def load_data():
    data = fetch_california_housing(as_frame=True)
    return data.data, data.target

def preprocess_data(X):
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    return X_scaled

# def train_and_log_model(model_name, model, X_train, X_test, y_train, y_test):
#     with mlflow.start_run(run_name=model_name):
#         model.fit(X_train, y_train)
#         y_pred = model.predict(X_test)

#         if model_name == "DecisionTree":
#             mlflow.log_param("max_depth", model.get_params()["max_depth"])

#         mlflow.log_param("model_type", model_name)

#         mse = mean_squared_error(y_test, y_pred)
#         mlflow.log_metric("mse", mse)

#         mlflow.sklearn.log_model(model, name="model")

#         print(f"Logged {model_name} with MSE: {mse:.4f}")

# def main():
#     X, y = load_data()
#     X_processed = preprocess_data(X)
#     X_train, X_test, y_train, y_test = train_test_split(X_processed, y, test_size=0.2, random_state=42)

#     models = {
#         "LinearRegression": LinearRegression(),
#         "DecisionTree": DecisionTreeRegressor(max_depth=5, random_state=42)
#     }
#     tracking_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "mlruns"))
#     tracking_uri = "file:///" + tracking_dir.replace("\\", "/")
#     print(f"Using tracking URI: {tracking_uri}")  
#     mlflow.set_tracking_uri(tracking_uri)
#     mlflow.set_experiment("California_Housing_Regression")
#     #mlflow.set_experiment("MLOPS - California_Housing_Regression Tracking")

#     for name, model in models.items():
#         train_and_log_model(name, model, X_train, X_test, y_train, y_test)



def train_and_log_model(model_name, model, X_train, X_test, y_train, y_test):
    with mlflow.start_run(run_name=model_name):
        model.fit(X_train, y_train)
        y_pred = model.predict(X_test)

        # Log model parameters
        if model_name == "DecisionTree":
            mlflow.log_param("max_depth", model.get_params()["max_depth"])

        mlflow.log_param("model_type", model_name)

        # Calculate and log MSE
        mse = mean_squared_error(y_test, y_pred)
        mlflow.log_metric("mse", mse)

        # Log the model
        mlflow.sklearn.log_model(model, "model")

        # Log test data and predictions as a CSV artifact
        df = pd.DataFrame(X_test)
        df["actual"] = y_test.reset_index(drop=True)
        df["predicted"] = y_pred

        with tempfile.NamedTemporaryFile(mode='w', suffix=".csv", delete=False) as tmp_file:
            df.to_csv(tmp_file.name, index=False)
            mlflow.log_artifact(tmp_file.name, artifact_path="dataset")

        print(f"Logged {model_name} | MSE: {mse:.4f} | Run ID: {mlflow.active_run().info.run_id}")





def main():
    X, y = load_data()
    X_processed = preprocess_data(X)
    X_train, X_test, y_train, y_test = train_test_split(X_processed, y, test_size=0.2, random_state=42)

    models = {
        "LinearRegression": LinearRegression(),
        "DecisionTree": DecisionTreeRegressor(max_depth=5, random_state=42)
    }

    tracking_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "mlruns"))
    tracking_uri = "file:///" + tracking_dir.replace("\\", "/")
    print(f"Using tracking URI: {tracking_uri}")  

    mlflow.set_tracking_uri(tracking_uri)
    mlflow.set_experiment("California_Housing_Regression")

    for name, model in models.items():
        train_and_log_model(name, model, X_train, X_test, y_train, y_test)


if __name__ == "__main__":
    print("run start" )
    main()
    print("run end" )
