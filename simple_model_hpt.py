import mlflow
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report
from tqdm import tqdm

from data_processing import get_data

mlflow.set_experiment("CovType")

with mlflow.start_run():
    mlflow.log_param("model_type", "Logistic Regression")

    max_iters = [10, 20, 50, 100, 200]

    mlflow.log_param("iters", max_iters)

    x_train, x_test, y_train, y_test, target_names = get_data()

    for iters in tqdm(max_iters):
        with mlflow.start_run(run_name=f"{iters}", nested=True):
            mlflow.sklearn.autolog()

            classifier = LogisticRegression(random_state=42, max_iter=iters, n_jobs=-1)
            classifier.fit(x_train, y_train)

            y_pred = classifier.predict(x_test)
            report = classification_report(y_test, y_pred, output_dict=True)
            mlflow.log_metrics(
                {
                    "test_weighted_avg_f1_score": report.get("weighted avg").get(
                        "f1-score"
                    ),
                    "test_weighted_avg_precision": report.get("weighted avg").get(
                        "precision"
                    ),
                    "test_weighted_avg_recall": report.get("weighted avg").get(
                        "recall"
                    ),
                }
            )
