from sklearn.ensemble import RandomForestClassifier
from src.misc.runner import run_alone
from src.misc.make_argparser import make_argparser_alone

parser = make_argparser_alone()


def rf_model(X_train, y_train):
    model = RandomForestClassifier(
        n_estimators=10, max_depth=10, random_state=42, n_jobs=1  # 10  # 10
    )
    model.fit(X_train, y_train)
    return model


def rf_main(args):
    if not (args.binary or args.multiclass):
        run_alone("dataset_binaire.csv", rf_model, args)
        run_alone("dataset_multiclass.csv", rf_model, args)
    if args.binary:
        run_alone("dataset_binaire.csv", rf_model, args)
    if args.multiclass:
        run_alone("dataset_multiclass.csv", rf_model, args)
        
if __name__ == "__main__":
    args = parser.parse_args()
    rf_main(args)
