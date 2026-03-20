import argparse

def make_argparser_alone():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "-b", "--binary", action="store_true", help="Train model for binary classification"
    )
    parser.add_argument(
        "-m",
        "--multiclass",
        action="store_true",
        help="Train model for multiclass classification",
    )
    parser.add_argument(
        "-c",
        "--confusion-matrix",
        action="store_true",
        help="Show the confusion matrix at the end of training",
    )
    parser.add_argument(
        "-s",
        "--save",
        action="store_true",
        help="Flag to save or not the model on the computer",
    )
    return parser

def make_main_argparser():
    parser = argparse.ArgumentParser()
    group1 = parser.add_mutually_exclusive_group()
    group1.add_argument(
        "--metrics",
        nargs="+",
        default=[],
        choices=["accuracy", "precision", "recall", "f1_score", "file_size","all"],
        help="List of metrics to display. If no metrics is specified, every metric will be display",
    )
    group1.add_argument("--make-datasets",action="store_true",help="generate the training dataset for both classification")
    group2 = parser.add_mutually_exclusive_group()
    group2.add_argument(
        "-b",
        "--binary",
        action="store_true",
        help="When on, display only binary classification",
    )
    group2.add_argument(
        "-m",
        "--multiclass",
        action="store_true",
        help="When on, display only multiclass classfication",
    )
    ## Model subparser
    subparser = parser.add_subparsers(help="Subcommand of model")
    model_parser = subparser.add_parser("model", help="Run only one model")
    model_parser.add_argument(
        "model", choices=["RF", "DT", "KNN", "DBSCAN"], help="Model to run"
    )
    model_parser.add_argument(
        "-b", "--binary", action="store_true", help="Train model for binary classification"
    )
    model_parser.add_argument(
        "-m",
        "--multiclass",
        action="store_true",
        help="Train model for multiclass classification",
    )
    model_parser.add_argument(
        "-c",
        "--confusion-matrix",
        action="store_true",
        help="Show the confusion matrix at the end of training",
    )
    model_parser.add_argument(
        "-s",
        "--save",
        action="store_true",
        help="Flag to save or not the model on the computer",
    )
    return parser