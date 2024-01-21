"""Module for storing namespaces with data."""

from types import SimpleNamespace

column_names_ns = SimpleNamespace(
    BEST_MODEL="BEST_MODEL",
    ENDO_COLUMNS=["PM10", "PM25", "PM1"],
    FEATURES="feature",
    FORECAST_COLUMNS=["VALUE", "time", "MODEL", "PREDICTED"],
    FILE_NAME="FILE_NAME",
    ID="ID",
    LAT="lat",
    LON="lon",
    MODEL="MODEL",
    PM10="PM10",
    PM25="PM25",
    PM1="PM1",
    PREDICTED="PREDICTED",
    SEPARATOR="__",
    TIME="time",
    VALUE="VALUE",
    X="x",
    Y="y",
)

data_ns = SimpleNamespace(
    DISTANCE_MATRIX_GRID_FILE="frozen_distance_matrix_grid.pkl",
    EVALUATION_PATH="evaluation.csv",
    FEATURES_PATH="features.csv",
    FORECAST_RESULT_DIR="forecast_result",
    GRID_FILE_PATH="grid/grid500x300.shp",
    ID_FILE_PATH="ids.txt",
    LAT_LON_FILE_PATH="instalations.csv",
    ONE_DATA_FILE="one_data_file.csv",
    SELECTION_FILE="selection.csv",
    SENSOR_DATA_DIR="sensors_data",
    TRANSFORMED_DATA_DIR="transformed_data",
    WEATHER_DATA_DIR="weather_data",
)

date_formats_ns = SimpleNamespace(
    MAX_TIME_FORMAT="%Y-%m-%d %H",
    MIN_TIME_FORMAT="%Y-%m-%d %H",
    SENSOR_DATETIME_FORMAT="%Y-%m-%dT%H:%M:%S.%fZ",
    WEATHER_DATETIME_FORMAT="%Y-%m-%dT%H:%M",
)

sensor_json_ns = SimpleNamespace(
    HISTORY="history",
    INDEXES="indexes",
    NAME="name",
    STANDARDS="standards",
    TILL_DATE_TIME="tillDateTime",
    VALUES="values",
    VALUE="value",
)

values_ns = SimpleNamespace(
    AVERAGE_MODEL="average_model",
)

weather_json_ns = SimpleNamespace(
    HOURLY="hourly",
)
