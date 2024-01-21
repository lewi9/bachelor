"""Module for DistanceMatrixTask."""
import logging
import pickle
from typing import Iterable, Literal, Union

import geopandas as gpd
import pandas as pd
from scipy.spatial import distance

from app.data_managers.namespaces import column_names_ns
from app.interpolation.task_interpolation import TaskInterpolation

METRIC = {
    "braycurtis": distance.braycurtis,
    "canberra": distance.canberra,
    "chebyshev": distance.chebyshev,
    "cityblock": distance.cityblock,
    "correlation": distance.correlation,
    "cosine": distance.cosine,
    "euclidean": distance.euclidean,
    "mahalanobis": distance.mahalanobis,
    "minkowski": distance.minkowski,
    "seuclidean": distance.seuclidean,
    "sqeuclidean": distance.sqeuclidean,
}


class _DataStructure:
    """
    Data class used to store distance matrix data.
    """

    def __init__(
        self,
        x: float,
        y: float,
    ):
        self.x = x
        self.y = y
        self.distance_matrix_dict = {}

    def update_distance_matrix(self, key: str, value: pd.Series):
        """Update distance matrix."""
        self.distance_matrix_dict |= {key: value}


class DistanceMatrixTask(TaskInterpolation):
    """Task for creating distance matrix."""

    def __init__(
        self,
        grid_path: str,
        stations_path: str,
        output_path: str,
        grid_epsg: int,
        metrics: Union[Iterable, Literal["ALL"]] = ("euclidean",),
    ):
        """
        Initialize DistanceMatrixTask.

        Parameters
        ----------
        grid_path : str
            Path to grid file. Grid file should be a shapefile.
        stations_path : str
            Path to stations file. It should be a csv file with lat, lon columns,
            epgs:4326 is expected.
        output_path : str
            Path to output file. It will be a pickled frozenset of _DataStructure objects.
        grid_epsg : int
            EPSG code of grid file.
        metrics : Iterable, optional
            Iterable of metrics to calculate, by default ("euclidean",)
        """
        super().__init__()
        self.grid_path = grid_path
        self.stations_path = stations_path
        self.output_path = output_path
        self.grid_epsg = grid_epsg

        if isinstance(metrics, str) and metrics == "ALL":
            metrics = METRIC.keys()
        elif isinstance(metrics, str):
            metrics = (metrics,)

        wrong_metrics = set(metrics).difference(METRIC.keys())
        if wrong_metrics:
            raise ValueError(
                f"Wrong metrics: {wrong_metrics}" f"Available metrics: {METRIC.keys()}"
            )
        self.metrics = metrics

    def _run(self):
        """
        Create distance matrix with given metrics. Save it to output_path.
        The output is pickled frozenset of _DataStructure objects.
        """
        grid_points = gpd.read_file(self.grid_path)
        set_of_grid_points = {
            _DataStructure(x=row[column_names_ns.X], y=row[column_names_ns.Y])
            for _, row in grid_points.iterrows()
        }

        stations = pd.read_csv(self.stations_path)
        station_points = gpd.GeoDataFrame(
            stations,
            geometry=gpd.points_from_xy(
                stations[column_names_ns.LON], stations[column_names_ns.LAT]
            ),
            crs="EPSG:4326",
        )
        station_points = station_points.to_crs(epsg=self.grid_epsg)
        station_points[
            [column_names_ns.X, column_names_ns.Y]
        ] = station_points.geometry.apply(lambda geom: pd.Series([geom.x, geom.y]))
        station_points = station_points.set_index(column_names_ns.ID, drop=True)
        station_points = station_points[[column_names_ns.X, column_names_ns.Y]]

        result = self._calc_distance_matrix(
            set_of_grid_points=set_of_grid_points, station_points=station_points
        )
        result = frozenset(result)

        with open(self.output_path, "wb") as file:
            pickle.dump(result, file)

    def _calc_distance_matrix(
        self, set_of_grid_points: set, station_points: gpd.GeoDataFrame
    ) -> set:
        """Calculate distance matrix for given grid points and stations."""
        for metric in self.metrics:
            logging.info("Calculating distance matrix for %s metric.", metric)
            for grid_point in set_of_grid_points:
                distances = self._calc_distance(
                    x=grid_point.x,
                    y=grid_point.y,
                    metric=metric,
                    station_points=station_points,
                )
                grid_point.update_distance_matrix(metric, distances)
        return set_of_grid_points

    def _calc_distance(
        self, x: float, y: float, metric: str, station_points: gpd.GeoDataFrame
    ) -> pd.Series:
        """
        Calculate distance for given point and metric.
        Return series with distances, that are sorted.
        """
        return station_points.apply(
            lambda sensor: METRIC[metric](
                (x, y),
                (sensor.loc[column_names_ns.X], sensor.loc[column_names_ns.Y]),
            ),
            axis=1,
        ).sort_values(ascending=True)
