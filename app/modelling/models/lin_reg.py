"""Wrapper for sklearn's LinearRegression model."""

from sklearn.linear_model import LinearRegression

from app.modelling.models.model_wrapper import ModelWrapper


class LinReg(ModelWrapper):
    """Wrapper for sklearn's LinearRegression model."""

    def __init__(self, *args, **kwargs):
        """Initialize the model."""
        self.model = LinearRegression(*args, **kwargs)
        self.indexes = None
