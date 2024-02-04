"""Module for regression tree model wrapper."""

from sklearn.tree import DecisionTreeRegressor

from app.modelling.models.model_wrapper import ModelWrapper


class RegressionTree(ModelWrapper):
    """Wrapper for sklearn's DecisionTreeRegressor model."""

    def __init__(self, *args, **kwargs):
        """Initialize the model."""
        self.model = DecisionTreeRegressor(*args, **kwargs)
        self.indexes = None
