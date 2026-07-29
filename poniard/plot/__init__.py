def __getattr__(name):
    if name == "PoniardPlotFactory":
        from poniard.plot.plot_factory import PoniardPlotFactory

        return PoniardPlotFactory
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")


__all__ = ["PoniardPlotFactory"]
