from src.misc.make_argparser import make_main_argparser
import pandas as pd
from src.misc.make_dataset import generate_datasets
from src.misc.helpers import train_test_data
from src.misc.metrics import generate_metrics
from src.random_forest import rf_model, rf_main
from src.misc.runner import run_for_metrics
from src.decision_tree import dt_model, dt_main
from src.knn import knn_model, knn_main
from src.dbscan import dbscan_model, dbscan_main
import numpy as np
import matplotlib.pyplot as plt
## PARSER ARGUMENTS
parser = make_main_argparser()


MODELS = {"RF": rf_model, "DT": dt_model, "KNN": knn_model, "DBSCAN": dbscan_model}
VERBOSITY = {
    "RF": "Random Forest",
    "DT": "Decision Tree",
    "KNN": "K-Nearest Neighbors",
    "DBSCAN": "DBSCAN",
}

def main_one(metrics:list[str], class_type:str):
    file_name = (
            "dataset_binaire.csv"
            if class_type == "binaire"
            else "dataset_multiclass.csv"
    )
    print("Getting Data", end=" ")
    df = pd.read_csv(file_name)
    print(" -- DONE")

    print("Transform " + class_type + " Data to training and test sets", end=" ")
    X_train, X_test, y_train, y_test, _ = train_test_data(df)
    print(" -- DONE")

    print("Start of All model training :")
    pred_per_model = {}
    models = {}
    for key in MODELS.keys():
        print(f"  - {VERBOSITY[key]}", end=" ")
        y_pred,y_eval, model = run_for_metrics(MODELS[key], X_train, y_train, X_test,y_test)
        pred_per_model[key] = y_pred
        models[key] = model
        print("-- DONE")
    # last y_eval is always DBSCAN
    Y = generate_metrics(metrics,pred_per_model,models,y_test,y_eval) # dict Metrics => [ Value for each models ]
    X = MODELS.keys()
    width = 0.4
    x = np.arange(len(X))

    for metric in Y.keys():
        fig, ax = plt.subplots()
        ax.grid()
        ax.set_xlabel("Modèles")
        if metric == "file_size":
            barres = ax.bar(x - width / 2, np.array(Y[metric])/1000, width, label=class_type)
            ax.set_yscale("log")
            ax.set_ylabel(f"{metric[0].upper()+metric[1:]} (in Ko)")
        else:
            ax.set_ylabel(f"{metric[0].upper()+metric[1:]}")
            barres = ax.bar(x - width / 2, Y[metric], width, label=class_type)
        ax.set_xticks(x)
        ax.set_xticklabels(X)
        
        ax.legend()
        plt.show()

    exit(0)

def main_both(metrics:list[str]):
    print("Getting Data", end="")
    dfm = pd.read_csv("dataset_multiclass.csv")
    dfb = pd.read_csv("dataset_binaire.csv")
    print(" -- DONE")

    print("Transform Data to training and test sets", end="")
    Xm_train, Xm_test, ym_train, ym_test, _ = train_test_data(dfm)
    Xb_train, Xb_test, yb_train, yb_test, _ = train_test_data(dfb)
    print(" -- DONE")
    print("Start of All model training :")

    pred_per_bin_model = {}
    pred_per_mult_model = {}
    bin_models = {}
    mult_models = {}
    for key in MODELS.keys():
        print(f"  - {VERBOSITY[key]} -- Binary Classification", end=" ")
        y_pred_bin,y_eval_bin, model_bin = run_for_metrics(MODELS[key], Xb_train, yb_train, Xb_test,yb_test)
        pred_per_bin_model[key] = y_pred_bin
        bin_models[key] = model_bin
        print("-- DONE")
        print(f"  - {VERBOSITY[key]} -- Multiclass Classification", end=" ")
        y_pred_mul,y_eval_mul, model_mul = run_for_metrics(MODELS[key], Xm_train, ym_train, Xm_test,ym_test)
        pred_per_mult_model[key] = y_pred_mul
        mult_models[key] = model_mul
        print("-- DONE")
    # last y_eval is always DBSCAN
    Y_bin = generate_metrics(metrics,pred_per_bin_model,bin_models,yb_test,y_eval_bin) # dict Metrics => [ Value for each models ]
    Y_mul = generate_metrics(metrics,pred_per_mult_model,mult_models,ym_test,y_eval_mul)
    X = MODELS.keys()
    width = 0.4
    x = np.arange(len(X))
    for metric in Y_bin.keys():
        fig, ax = plt.subplots()
        ax.grid()
        ax.set_xlabel("Modèles")
        ax.set_ylabel(f"{metric[0].upper()+metric[1:]}")
        ax.set_xticks(x)
        ax.set_xticklabels(X)
        if metric == "file_size":
            barres_bin = ax.bar(x - width / 2, np.array(Y_bin[metric])/1000, width, label="Binary")
            barres_mul = ax.bar(x + width/2, np.array(Y_mul[metric])/1000,width, label="Multiclass")
            ax.set_yscale("log")
            ax.set_ylabel(f"{metric[0].upper()+metric[1:]} (in Ko)")
        else:
            ax.set_ylabel(f"{metric[0].upper()+metric[1:]}")
            barres_bin = ax.bar(x - width / 2, Y_bin[metric], width, label="Binary")
            barres_mul = ax.bar(x + width/2, Y_mul[metric],width, label="Multiclass")
        ax.legend()
        plt.show()

    exit(0)
    


def main(metrics:list[str], class_type:str):
    if len(metrics) == 0:
        metrics.append("all")

    if class_type is None:
        main_both(metrics)
    # or else we go with the given class_type
    else:
        main_one(metrics,class_type)
        

MODELS_MAIN ={
    "RF" : rf_main,
    "DT" : dt_main,
    "KNN" : knn_main,
    "DBSCAN" : dbscan_main
}

if __name__ == "__main__":
    args = parser.parse_args()
    if args.make_datasets:
        generate_datasets()
        exit(0)
    # go only for one model
    if hasattr(args, 'model'):
        print(f"Going with model : {args.model}")
        MODELS_MAIN[args.model](args)
        exit(0)
        
    # Generate metrics from every models and display them
    if args.binary:
        main(args.metrics, "binary")
    elif args.multiclass:
        main(args.metrics, "multiclass")
    else:
        main(args.metrics, None)


