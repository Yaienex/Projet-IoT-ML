from sklearn.metrics import (
    accuracy_score,
    classification_report,
    precision_score,
    recall_score,
    f1_score,
    confusion_matrix,
)
import matplotlib.pyplot as plt
import seaborn as sns
import os
from joblib import dump

def print_metrics(y_pred, y_test):
    accuracy = accuracy_score(y_test, y_pred)
    print(f"Accuracy: {accuracy*100:.2f}%")

    precision = precision_score(y_test, y_pred, average="weighted")
    print(f"Precision: {precision:.4f}")

    recall = recall_score(y_test, y_pred, average="weighted")
    print(f"Recall: {recall:.4f}")

    f1 = f1_score(y_test, y_pred, average="weighted")
    print(f"F1-score: {f1:.4f}")


def show_confusion_matrix(y_pred, y_test, le):
    cm = confusion_matrix(y_test, y_pred)
    plt.figure(figsize=(10, 7))
    sns.heatmap(
        cm,
        annot=True,
        fmt="d",
        cmap="Blues",
        xticklabels=le.classes_,
        yticklabels=le.classes_,
    )
    plt.title("Matrice de confusion")
    plt.xlabel("Prédit")
    plt.ylabel("Vrai")
    plt.show()


def calculate_accuracy_score(y_pred,y_test,model=None):
    return accuracy_score(y_pred,y_test)
def calculate_precision_score(y_pred,y_test,model=None):
    return precision_score(y_pred,y_test,  average="weighted")
def calculate_recall_score(y_pred,y_test,model=None):
    return recall_score(y_pred,y_test,  average="weighted")
def calculate_f1_score(y_pred,y_test,model=None):
    return f1_score(y_pred,y_test,  average="weighted")
def calculate_file_size(y_pred,y_test,model=None):
    dump(model,"model.joblib")
    file_size = os.path.getsize('model.joblib')
    os.remove("model.joblib")
    return file_size

FUNCTIONS_DIR = {
    "accuracy" : calculate_accuracy_score,
    "precision" : calculate_precision_score,
    "recall" : calculate_recall_score,
    "f1_score" : calculate_f1_score,
    "file_size" : calculate_file_size
}
MODELS_ORDER = ["RF","DT","KNN","DBSCAN"]
ALL_METRICS = ["accuracy","precision","recall","f1_score","file_size"]

def generate_metrics(metrics, pred_per_model,models, y_test,y_eval=None):
    # Always in the same order : RF DT KNN DBSCAN
    metrics_holder = { m : [] for m in metrics} if "all" not in metrics else {m:[] for m in ALL_METRICS}
    if "all" in metrics:
        for metric in ALL_METRICS:
            for model in MODELS_ORDER:
                if model == "DBSCAN":
                    metrics_holder[metric].append(FUNCTIONS_DIR[metric](pred_per_model[model],y_eval,models[model]))
                else:
                    metrics_holder[metric].append(FUNCTIONS_DIR[metric](pred_per_model[model],y_test,models[model]))
    else :
        for metric in metrics:
            for model in MODELS_ORDER:
                if model == "DBSCAN":
                    metrics_holder[metric].append(FUNCTIONS_DIR[metric](pred_per_model[model],y_eval,models[model]))
                else:
                    metrics_holder[metric].append(FUNCTIONS_DIR[metric](pred_per_model[model],y_test,models[model]))


    return metrics_holder