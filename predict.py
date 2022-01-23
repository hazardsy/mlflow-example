import mlflow
from data_processing import get_data

model_name = "CovType"

model = mlflow.pyfunc.load_model(f"models:/{model_name}/latest")

x_train, x_test, y_train, y_test, target_names = get_data()

predictions = model.predict(x_test)

print(predictions)
