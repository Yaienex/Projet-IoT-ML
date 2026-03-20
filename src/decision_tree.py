from sklearn.tree import DecisionTreeClassifier
from src.misc.runner import run_alone
from src.misc.make_argparser import make_argparser_alone
parser = make_argparser_alone()

def dt_model(X_train, y_train):
    model = DecisionTreeClassifier(
        max_depth=20,  # multi 20 Gini
        random_state=42,  # binary 10 entropy
        criterion="gini",
        min_samples_split=100,
    )
    model.fit(X_train, y_train)
    return model

def dt_main(args):
    # no classification specify, so we run both
    if not (args.binary or args.multiclass):
        run_alone("dataset_binaire.csv", dt_model, args)
        run_alone("dataset_multiclass.csv", dt_model, args)
    if args.binary:
        run_alone("dataset_binaire.csv", dt_model, args)
    if args.multiclass:
        run_alone("dataset_multiclass.csv", dt_model, args)

if __name__ == "__main__":
    args = parser.parse_args()
    dt_main(args)
