from sklearn.cluster import DBSCAN
from src.misc.runner import run_dbscan
from src.misc.make_argparser import make_argparser_alone
parser = make_argparser_alone()

def dbscan_model(X_train):
    eps = 0.01
    model = DBSCAN(eps=eps, min_samples=10)
    train_labels = model.fit_predict(X_train)
    return train_labels, model

def dbscan_main(args):
    # no classification specify, so we run both
    if not (args.binary or args.multiclass):
        run_dbscan("dataset_binaire.csv", dbscan_model, args)
        run_dbscan("dataset_multiclass.csv", dbscan_model, args)
    if args.binary:
        run_dbscan("dataset_binaire.csv", dbscan_model, args)
    if args.multiclass:
        run_dbscan("dataset_multiclass.csv", dbscan_model, args)

if __name__ == "__main__":
    args = parser.parse_args()
    dbscan_main(args)
