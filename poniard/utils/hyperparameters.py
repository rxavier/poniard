import numpy as np

GRID = {
    "LogisticRegression": {"C": [0.1, 1, 10, 100], "penalty": ["l1", "l2", "none"]},
    "SVC": {
        "kernel": ["linear", "rbf"],
        "C": [0.1, 1, 10, 100],
    },
    "KNeighborsClassifier": {
        "n_neighbors": list(range(1, 12, 2)),
        "weights": ["uniform", "distance"],
        "leaf_size": list(range(10, 51, 10)),
    },
    "DecisionTreeClassifier": {
        "criterion": ["gini", "entropy"],
        "min_samples_split": list(range(2, 21, 2)),
    },
    "RandomForestClassifier": {
        "criterion": ["gini", "entropy"],
        "min_samples_split": list(range(2, 21, 2)),
        "n_estimators": [50] + list(range(100, 501, 100)),
    },
    "HistGradientBoostingClassifier": {
        "max_iter": [50] + list(range(100, 501, 100)),
        "learning_rate": [0.01, 0.1, 0.5, 0.9],
        "min_samples_leaf": list(range(10, 101, 10)),
    },
    "XGBClassifier": {
        "learning_rate": [0.01, 0.1, 0.3, 0.5, 0.9],
        "max_depth": list(range(3, 16, 3)),
        "min_child_weight": list(range(1, 8, 2)),
        "colsample_bytree": [0.3, 0.4, 0.5, 0.7],
    },
    "ElasticNet": {"alpha": [0.5, 1], "l1_ratio": np.arange(0.1, 1, 0.2)},
    "SVR": {
        "kernel": ["linear", "rbf"],
        "C": [0.1, 1, 10, 100],
    },
    "KNeighborsRegressor": {
        "n_neighbors": list(range(1, 12, 2)),
        "weights": ["uniform", "distance"],
        "leaf_size": list(range(10, 51, 10)),
    },
    "DecisionTreeRegressor": {
        "criterion": ["squared_error", "friedman_mse"],
        "min_samples_split": list(range(2, 21, 2)),
    },
    "RandomForestRegressor": {
        "criterion": ["squared_error", "friedman_mse"],
        "min_samples_split": list(range(2, 21, 2)),
        "n_estimators": [50] + list(range(100, 501, 100)),
    },
    "HistGradientBoostingRegressor": {
        "max_iter": [50] + list(range(100, 501, 100)),
        "learning_rate": [0.01, 0.1, 0.5, 0.9],
        "min_samples_leaf": list(range(10, 101, 10)),
    },
    "XGBRegressor": {
        "learning_rate": [0.01, 0.1, 0.3, 0.5, 0.9],
        "max_depth": list(range(3, 16, 3)),
        "min_child_weight": list(range(1, 8, 2)),
        "colsample_bytree": [0.3, 0.4, 0.5, 0.7],
    },
}


def get_grid(model_name: str) -> dict:
    """Obtain a default parameter grid."""
    return GRID[model_name]
