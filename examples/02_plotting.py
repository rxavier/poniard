"""Plotting with PoniardPlotFactory (requires: pip install poniard[plot])."""

from sklearn.datasets import make_classification
from sklearn.linear_model import LogisticRegression

from poniard import PoniardClassifier
from poniard.plot import PoniardPlotFactory

X, y = make_classification(n_samples=200, n_features=8, random_state=42)

clf = PoniardClassifier(estimators=[LogisticRegression()])
clf.fit(X, y, show_info=False)

# The plot factory is a standalone object: it never mutates the estimator or
# plotly's global state.
plotter = PoniardPlotFactory(X, y, clf)

metrics_fig = plotter.metrics()
roc_fig = plotter.roc_curve()
cm_fig = plotter.confusion_matrix("LogisticRegression")
importance_fig = plotter.permutation_importance("LogisticRegression", n_repeats=5)
time_quality_fig = plotter.time_quality_scatter()

# Single-figure dashboard: metric rankings, model comparison, ROC/confusion
# matrix and permutation importance in one view.
dashboard = plotter.full_estimator_analysis("LogisticRegression")

print("Built figures:")
for fig in (metrics_fig, roc_fig, cm_fig, importance_fig, time_quality_fig, dashboard):
    print(f"  - {fig.layout.title.text if fig.layout.title else type(fig).__name__}")

if __name__ == "__main__":
    dashboard.show()
