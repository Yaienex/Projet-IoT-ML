from sklearn.neighbors import KNeighborsClassifier
from src.misc.runner import run_alone
from src.misc.make_argparser import make_argparser_alone

parser = make_argparser_alone()

def knn_model(X_train, y_train):
    model = KNeighborsClassifier(
        n_neighbors=10, weights="uniform", metric="minkowski", p=2  # p = 2 = euclid
    )
    model.fit(X_train, y_train)
    return model

def knn_main(args):

    # no classification specify, so we run both
    if not (args.binary or args.multiclass):
        run_alone("dataset_binaire.csv", knn_model, args)
        run_alone("dataset_multiclass.csv", knn_model, args)
    if args.binary:
        run_alone("dataset_binaire.csv", knn_model, args)
    if args.multiclass:
        run_alone("dataset_multiclass.csv", knn_model, args)

if __name__ == "__main__":
    args = parser.parse_args()
    knn_main(args)
