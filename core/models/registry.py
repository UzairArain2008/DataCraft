from sklearn.linear_model import LogisticRegression, LinearRegression
from sklearn.neighbors import KNeighborsClassifier, KNeighborsRegressor
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.svm import SVC, SVR
from sklearn.cluster import KMeans, DBSCAN

MODEL_REGISTRY = {
    "classification": {
        "logistic_regression": LogisticRegression,
        "knn": KNeighborsClassifier,
        "random_forest": RandomForestClassifier,
        "svm": SVC,
    },
    "regression": {
        "linear_regression": LinearRegression,
        "knn": KNeighborsRegressor,
        "random_forest": RandomForestRegressor,
        "svr": SVR,
    },
    "clustering": {
        "kmeans": KMeans,
        "dbscan": DBSCAN,
    }
}
