import pandas as pd
from sklearn.metrics import classification_report
from joblib import dump

from src.misc.metrics import  show_confusion_matrix, print_metrics
from src.misc.helpers import dbscan_predict, train_test_data


def run_alone(df_name, generate_model, args):
    print("Getting Data", end=" ")
    df = pd.read_csv(df_name)
    print("-- Done")

    print("Transform Data to training and test sets", end=" ")
    X_train, X_test, y_train, y_test, le = train_test_data(df)
    print("-- Done")

    print("Training model", end=" ")
    model = generate_model(X_train, y_train)
    print("-- Done")

    print("Prediction", end=" ")
    y_pred = model.predict(X_test)
    print("-- Done\n")

    print("Results : ")
    print_metrics(y_pred, y_test)
    print(classification_report(y_test, y_pred, target_names=le.classes_))

    if args.confusion_matrix:
        show_confusion_matrix(y_pred, y_test, le)

    if args.save:
        name = generate_model.__name__ + "_" + df_name[8:-4] + ".joblib"
        print("Model save at : ", name)
        dump(model, name)


def run_dbscan(df_name, generate_model, args):
    print("Getting Data", end=" ")
    df = pd.read_csv(df_name)
    print("-- Done")

    print("Transform Data to training and test sets", end=" ")
    X_train, X_test, y_train, y_test, le = train_test_data(df)
    print("-- Done")

    print("Training model", end=" ")
    train_pred, model = generate_model(X_train)
    print("-- Done")

    print("Prediction", end=" ")
    y_eval, y_pred = dbscan_predict(X_train, X_test, train_pred, y_test)
    print("-- Done\n")

    print("Results : ")
    print_metrics(y_pred, y_eval)
    print(classification_report(y_eval, y_pred, target_names=le.classes_))

    if args.confusion_matrix:
        show_confusion_matrix(y_pred, y_eval, le)

    if args.save:
        name = generate_model.__name__ + "_" + df_name[8:-4] + ".joblib"
        print("Model save at : ", name)
        dump(model, name)


def run_for_metrics(generate_model, X_train, y_train, X_test,y_test):
    if generate_model.__name__  == "dbscan_model":
        train_pred, model = generate_model(X_train)
        y_eval, y_pred = dbscan_predict(X_train, X_test, train_pred, y_test)
        return y_pred, y_eval, model
    else:
        model = generate_model(X_train, y_train)
        y_pred = model.predict(X_test)
        return y_pred, None, model
