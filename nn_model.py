import mlflow
from sklearn.metrics import classification_report
from sklearn.neural_network import MLPClassifier

from data_processing import get_data

mlflow.set_experiment("CovType")

with mlflow.start_run():
    mlflow.log_param("model_type", "Multilayer Perceptron")
    mlflow.sklearn.autolog()

    x_train, x_test, y_train, y_test, target_names = get_data()

    classifier = MLPClassifier(
        hidden_layer_sizes=(10, 20), random_state=42, verbose=True, early_stopping=True
    )

    classifier.fit(x_train, y_train)

    y_pred = classifier.predict(x_test)

    report = classification_report(y_test, y_pred, output_dict=True)
    mlflow.log_metrics(
        {
            "test_weighted_avg_f1_score": report.get("weighted avg").get("f1-score"),
            "test_weighted_avg_precision": report.get("weighted avg").get("precision"),
            "test_weighted_avg_recall": report.get("weighted avg").get("recall"),
        }
    )
