import subprocess
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression

from poniard import PoniardClassifier
from poniard.preprocessing import PoniardPreprocessor


def test_importing_poniard_does_not_change_sklearn_global_config():
    """Importing poniard must not mutate sklearn's global ``transform_output`` config."""
    subprocess.run(
        [
            sys.executable,
            "-c",
            "import sklearn; sklearn.set_config(transform_output='default'); "
            "import poniard; "
            "from sklearn import get_config; "
            "assert get_config()['transform_output'] == 'default', get_config()",
        ],
        check=True,
    )


def test_preprocessor_outputs_pandas_without_global_set_config():
    """The preprocessor's Pipeline must return pandas DataFrames on its own,
    without depending on a global sklearn set_config call."""
    from sklearn import set_config

    set_config(transform_output="default")
    try:
        preprocessor = PoniardPreprocessor(task="classification")
        X = pd.DataFrame(
            {
                "A": [4.0, 3.0, 1.0, -1.0, np.nan],
                "B": [-2.0, np.nan, 3.0, 7.0, 1.0],
                "C": list("abcde"),
            }
        )
        y = pd.Series([0, 1, 0, 1, 0])
        preprocessor.build(X=X, y=y, task="classification")
        preprocessor.preprocessor.fit(X, y)
        transformed = preprocessor.preprocessor.transform(X)
        assert isinstance(transformed, pd.DataFrame)
    finally:
        set_config(transform_output="default")


def test_cache_transformations_does_not_write_to_cwd(tmp_path, monkeypatch):
    """``cache_transformations=True`` must not create a directory in the current
    working directory; the cache should live under a temp dir by default."""
    monkeypatch.chdir(tmp_path)
    preprocessor = PoniardPreprocessor(
        task="classification",
        cache_transformations=True,
    )
    preprocessor.build(
        X=pd.DataFrame({"A": [1.0, 2.0, 3.0, 4.0, 5.0]}), y=pd.Series([0, 1, 0, 1, 0])
    )
    cwd_contents = list(tmp_path.iterdir())
    assert cwd_contents == [], f"Files were created in CWD: {cwd_contents}"
    assert preprocessor._memory is not None
    assert preprocessor._cache_tempdir is not None
    cache_path = Path(preprocessor._cache_tempdir.name)
    assert cache_path.exists()
    # cleanup happens automatically on garbage collection; force it here
    del preprocessor
    import gc

    gc.collect()
    assert not cache_path.exists()


def test_cache_transformations_with_explicit_cache_dir(tmp_path):
    """A user-provided ``cache_dir`` is honored and the user owns its contents."""
    user_dir = tmp_path / "my_cache"
    preprocessor = PoniardPreprocessor(
        task="classification",
        cache_transformations=True,
        cache_dir=user_dir,
    )
    preprocessor.build(
        X=pd.DataFrame({"A": [1.0, 2.0, 3.0, 4.0, 5.0]}), y=pd.Series([0, 1, 0, 1, 0])
    )
    assert preprocessor._cache_tempdir is None
    assert preprocessor._memory is not None
    assert preprocessor._memory.location == str(user_dir)
    assert user_dir.exists()


def test_cache_transformations_false_means_no_memory():
    preprocessor = PoniardPreprocessor(
        task="classification",
        cache_transformations=False,
    )
    assert preprocessor._memory is None
    assert preprocessor._cache_tempdir is None


def test_plot_factory_does_not_mutate_plotly_globals():
    """Constructing PoniardPlotFactory and rendering a figure must not mutate
    plotly's global template or default color sequence."""
    code = (
        "import plotly.express as px, plotly.io as pio; "
        "from poniard import PoniardClassifier; "
        "before_t = pio.templates.default; "
        "before_c = px.defaults.color_discrete_sequence; "
        "from poniard.plot import PoniardPlotFactory; "
        "import numpy as np, pandas as pd; "
        "from sklearn.linear_model import LogisticRegression; "
        "X = pd.DataFrame(np.random.normal(size=(30, 3)), columns=list('abc')); "
        "X['s'] = np.random.choice(['x','y','z'], size=30); "
        "y = np.random.choice([0,1], size=30); "
        "clf = PoniardClassifier(estimators=[LogisticRegression()], cv=2, random_state=0); "
        "clf.setup(X, y); clf.fit(X, y); "
        "plotter = PoniardPlotFactory(X, y, clf); "
        "plotter.metrics(); "
        "assert pio.templates.default == before_t, (pio.templates.default, before_t); "
        "assert px.defaults.color_discrete_sequence == before_c, "
        "(px.defaults.color_discrete_sequence, before_c); "
    )
    subprocess.run([sys.executable, "-c", code], check=True)


def test_plot_factory_applies_per_figure_template_and_colors():
    """The template and discrete colors passed at construction must end up on
    the returned figure, not on the plotly globals."""
    import plotly.graph_objects as go

    from poniard.plot import PoniardPlotFactory

    X = pd.DataFrame(np.random.normal(size=(30, 3)), columns=list("abc"))
    X["s"] = np.random.choice(["x", "y", "z"], size=30)
    y = np.random.choice([0, 1], size=30)
    clf = PoniardClassifier(estimators=[LogisticRegression()], cv=2, random_state=0)
    clf.setup(X, y)
    clf.fit(X, y)

    plotter = PoniardPlotFactory(
        X,
        y,
        clf,
        discrete_colors=["#112233", "#445566", "#778899"],
        font_family="Courier",
        font_color="#abcdef",
    )
    fig = plotter.metrics()
    assert isinstance(fig, go.Figure)
    assert fig.layout.font.family == "Courier"
    assert fig.layout.font.color == "#abcdef"
    # Legend orientation matches the original config
    assert fig.layout.legend.orientation == "h"


def test_custom_sklearn_preprocessor_outputs_pandas_without_global_set_config():
    """A user-supplied custom preprocessor (plain sklearn Pipeline) must also
    output pandas DataFrames when transformed, without depending on a global
    sklearn set_config call."""
    from sklearn import set_config
    from sklearn.base import clone
    from sklearn.impute import SimpleImputer
    from sklearn.pipeline import Pipeline
    from sklearn.preprocessing import StandardScaler

    set_config(transform_output="default")
    try:
        n = 50
        X = pd.DataFrame({"num": np.random.normal(size=n)})
        X.iloc[0, 0] = np.nan
        y = np.zeros(n, dtype=int)
        y[::2] = 1
        custom = Pipeline(
            [("imputer", SimpleImputer()), ("scaler", StandardScaler())]
        )
        clf = PoniardClassifier(
            estimators=[LogisticRegression()],
            custom_preprocessor=custom,
            preprocess=True,
            cv=2,
            random_state=0,
        )
        clf.setup(X, y)
        # Poniard must configure the user-supplied preprocessor to output
        # pandas so a fitted clone produces a DataFrame even when the global
        # sklearn config is "default".
        fitted = clone(clf.preprocessor)
        fitted.fit(X, y)
        out = fitted.transform(X)
        assert isinstance(out, pd.DataFrame)
    finally:
        set_config(transform_output="default")


def test_plot_factory_does_not_fit_stored_pipelines():
    """Plot methods that need a fitted model fit a clone, never the stored pipeline."""
    from sklearn.linear_model import LogisticRegression

    from poniard.plot import PoniardPlotFactory

    X = pd.DataFrame(np.random.normal(size=(40, 3)), columns=list("abc"))
    X["s"] = np.random.choice(["x", "y", "z"], size=40)
    y = pd.Series(np.random.choice([0, 1], size=40))
    clf = PoniardClassifier(estimators=[LogisticRegression()], cv=2, random_state=0)
    clf.setup(X, y)
    clf.fit(X, y)
    stored = clf.pipelines["LogisticRegression"]
    assert not hasattr(stored, "classes_")

    plotter = PoniardPlotFactory(X, y, clf)
    plotter.permutation_importance("LogisticRegression", n_repeats=1)
    assert not hasattr(clf.pipelines["LogisticRegression"], "classes_")

    plotter.partial_dependence("LogisticRegression", feature=0)
    assert not hasattr(clf.pipelines["LogisticRegression"], "classes_")
