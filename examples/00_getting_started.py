"""Fit, compare, and predict — the minimal Poniard loop."""

from sklearn.datasets import make_classification

from poniard import PoniardClassifier

X, y = make_classification(n_samples=100, n_features=10, random_state=42)

# Type inference, preprocessing, and cross-validation for every estimator,
# plus an automatic dummy baseline.
clf = PoniardClassifier()
clf.fit(X, y, show_info=False)

results = clf.get_results()
print("Leaderboard (mean test scores, sorted):")
print(results)

# Cross-validated predictions where each sample belongs to a single test fold.
predictions = clf.predict(X, y)
print(
    f"\nCV predictions for {len(predictions)} estimators, e.g. "
    f"{list(predictions)[0]}: {predictions[list(predictions)[0]].shape}"
)
