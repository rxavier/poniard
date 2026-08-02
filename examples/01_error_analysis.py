"""Error analysis: where and why your models fail."""

from sklearn.datasets import make_classification
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression

from poniard import PoniardClassifier
from poniard.error_analysis import ErrorAnalyzer

X, y = make_classification(n_samples=300, n_features=12, n_informative=6, random_state=42)

clf = PoniardClassifier(estimators=[LogisticRegression(), RandomForestClassifier(random_state=0)])
clf.fit(X, y, show_info=False)

# One call runs the full workflow and packages it into a structured report.
ea = ErrorAnalyzer.from_poniard(clf)  # all non-dummy estimators by default
report = ea.analyze(X, y)

print("Per-estimator summary:")
print(report.summary)

print(
    f"\nUniversal failures (every model got these wrong): {len(report.universal_failures)} samples"
)
print(f"Disagreement set (models split): {len(report.disagreement_set)} samples")

print("\nError lift by target (lift > 1 = class over-represented in errors):")
print(report.lift_by_target.head())

# The same steps, run individually.
ranked = ea.rank_errors(X, y)
merged = ErrorAnalyzer.merge_errors(ranked)
print("\nWorst samples across models (freq = # estimators that failed):")
print(merged.head(5))

print("\nError rate per target class:")
print(ea.analyze_target(errors_idx=merged.index, y=y))

print("\nTop features by permutation importance:")
top_features = ea.analyze_features(
    errors_idx=merged.index,
    X=X,
    y=y,
    estimator_name="LogisticRegression",
    n_features=3,
)
for feature, table in top_features.items():
    print(f"\n{feature}:")
    print(table)
