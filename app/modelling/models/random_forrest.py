"""Module for random forrest model wrapper."""

from sklearn.ensemble import RandomForestRegressor

from app.modelling.models.model_wrapper import ModelWrapper


class RandomForrest(ModelWrapper):
    """Wrapper for sklearn's RandomForestRegressor model."""

    def __init__(self, *args, **kwargs):
        """Initialize the model."""
        self.model = RandomForestRegressor(*args, **kwargs)
        self.indexes = None
