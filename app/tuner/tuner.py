"""Tuner module."""

import os
import pickle
from collections import OrderedDict
from typing import Iterable, Sequence, Union

import numpy as np
from pymoo.algorithms.moo.nsga2 import NSGA2
from pymoo.algorithms.moo.nsga3 import NSGA3
from pymoo.core.problem import Problem
from pymoo.optimize import minimize
from pymoo.util.ref_dirs import get_reference_directions

from app.modelling.agg_model_strategy_task import AggModelStrategyTask
from app.modelling.best_model_strategy_task import BestModelStrategyTask
from app.modelling.forecasting_pipeline_task import ForecastingPipelineTask
from app.utils.evaluator import Evaluator
from app.utils.process_pipeline import ProcessPipeline
from app.utils.task_pipeline import TaskPipeline

BEST_MODEL = "best_model"


class Tuner:
    """Tuner class."""

    def __init__(
        self,
        forecasters: dict[dict],
        splitters: dict,
        metrics: dict,
        strategies: dict,
        dates: Iterable[str],
        input_dir: str,
        result_dir: str,
        selection_dir: str,
        max_forecasts: int,
        transformers: Iterable,
        forecast_horizon: Union[int, Sequence[int], np.ndarray[int]],
        parallel_batch: int,
        engine: str,
    ):
        """
        Parameters
        ----------
        forecasters : dict[dict]
            Dictionary of dictionaries of forecasters.
        splitters : dict
            Dictionary of splitters.
        metrics : dict
            Dictionary of metrics.
        strategies : dict
            Dictionary of strategies.
        dates : Iterable[str]
            List of dates.
        input_dir : str
            Input directory.
        result_dir : str
            Result directory.
        selection_dir : str
            Selection directory.
        max_forecasts : int
            Maximum number of forecasts.
        transformers : Iterable
            List of transformers.
        forecast_horizon : Union[int, Sequence[int], np.ndarray[int]]
            Forecast horizon.
        parallel_batch : int
            Parallel batch.
        engine : str
            Engine to use.
        """
        self.forecasters = forecasters
        self.splitters = splitters
        self.metrics = metrics
        self.strategies = strategies
        self.dates = dates
        self.input_dir = input_dir
        self.result_dir = result_dir
        self.selection_dir = selection_dir
        self.max_forecasts = max_forecasts
        self.transformers = transformers
        self.forecast_horizon = forecast_horizon
        self.parallel_batch = parallel_batch
        self.engine = engine

    def tune(
        self,
        n_gen: int,
        n_pop: int,
        ev_metrics: dict,
        result_dir: str,
        time_constraint: int,
    ):
        """Tune the model."""
        os.mkdir(result_dir)
        problem = self._build_problem(
            ev_metrics=ev_metrics,
            result_dir=result_dir,
            time_constraint=time_constraint,
        )

        if self.engine == "nsga_3":
            directions = len(ev_metrics.keys()) + 1
            ref_dirs = get_reference_directions(
                "das-dennis", directions, n_partitions=2
            )

            algorithm = NSGA3(pop_size=n_pop, ref_dirs=ref_dirs)

            res = minimize(problem, algorithm, seed=1, termination=("n_gen", n_gen))

        if self.engine == "nsga_2":

            algorithm = NSGA2(pop_size=n_pop)

            res = minimize(problem, algorithm, ("n_gen", n_gen), seed=1, verbose=False)

        pickle.dump(res.F, open(os.path.join(result_dir, "final_F.pickle"), "wb"))
        pickle.dump(res.X, open(os.path.join(result_dir, "final_X.pickle"), "wb"))

    def _build_problem(
        self, ev_metrics: dict, result_dir: str, time_constraint: int
    ) -> Problem:
        """Build problem."""

        class TunerProblem(Problem):
            """Tuner problem."""

            def __init__(self, tuner: Tuner, ev_metrics: dict):
                self.tuner = tuner
                self.counter = 0
                n_obj = len(ev_metrics.keys()) * 3 + 2
                len_forecasters = len(tuner.forecasters.keys())
                len_splitters = len(tuner.splitters.keys())
                len_strategies = len(tuner.strategies.keys())
                len_metrics = len(tuner.metrics.keys())
                eps = 0.0001
                super().__init__(
                    n_var=4,
                    n_obj=n_obj,
                    xl=[1, 1, 0, 0],
                    xu=[
                        2**len_forecasters - eps,
                        2**len_splitters - eps,
                        len_strategies - eps,
                        len_metrics - eps,
                    ],
                    n_ieq_constr=1,
                )

            def _evaluate(self, x, out, *args, **kwargs):
                F = []
                G = []
                self.counter += 1
                for iteratation_, vec in enumerate(x):
                    sub_F = []
                    sub_G = []

                    forecasters_string = bin(int(vec[0]))[2:][::-1]
                    splitters_string = bin(int(vec[1]))[2:][::-1]
                    strategy = int(vec[2])
                    metric = int(vec[3])

                    forecasters = {}
                    splitters = {}

                    for i, package in enumerate(self.tuner.forecasters.keys()):
                        if i >= len(forecasters_string):
                            break
                        if forecasters_string[i] == "1":
                            forecasters |= self.tuner.forecasters[package]

                    for i, splitter in enumerate(self.tuner.splitters.keys()):
                        if i >= len(splitters_string):
                            break
                        if splitters_string[i] == "1":
                            splitters |= {splitter: self.tuner.splitters[splitter]}

                    if list(self.tuner.strategies.keys())[strategy] == BEST_MODEL:
                        for date in self.tuner.dates:
                            modelling_tasks = OrderedDict()
                            modelling_tasks |= {
                                "Forecasting Pipeline": ForecastingPipelineTask(
                                    input_dir=self.tuner.input_dir,
                                    result_dir=self.tuner.result_dir,
                                    max_forecasts=self.tuner.max_forecasts,
                                    transformers=self.tuner.transformers,
                                    forecasters=forecasters,
                                    splitters=splitters,
                                    last_valid_actual=date,
                                    exo_filler="mean",
                                    parallel_batch=self.tuner.parallel_batch,
                                )
                            }
                            modelling_tasks |= {
                                "Best Model Strategy": BestModelStrategyTask(
                                    transformed_data_dir=self.tuner.input_dir,
                                    forecast_dir=self.tuner.result_dir,
                                    result_dir=self.tuner.selection_dir,
                                    metric=list(self.tuner.metrics.values())[metric],
                                    last_evaluation_date=date,
                                    mode="recreate",
                                )
                            }
                            pipelines = OrderedDict()
                            pipelines |= {
                                "Modelling Pipeline": TaskPipeline(
                                    tasks=modelling_tasks
                                ),
                            }
                            process_pipeline = ProcessPipeline(pipelines=pipelines)
                            process_pipeline.run()
                            ev = Evaluator(
                                selected_data_dir=self.tuner.selection_dir,
                                reference_data_dir=self.tuner.input_dir,
                                metrics=list(ev_metrics.values()),
                            )
                            results = ev.evaluate()
                            results["execution_time"] = (
                                process_pipeline.execution_time.seconds
                            )
                            results["compared"] = -ev.compared
                            sub_F.append([*results.values()])
                            sub_G.append(
                                results["execution_time"] - time_constraint * 3600
                            )

                    else:
                        for date in self.tuner.dates:
                            modelling_tasks = OrderedDict()
                            modelling_tasks |= {
                                "Forecasting Pipeline": ForecastingPipelineTask(
                                    input_dir=self.tuner.input_dir,
                                    result_dir=self.tuner.result_dir,
                                    max_forecasts=1,
                                    transformers=self.tuner.transformers,
                                    forecasters=forecasters,
                                    splitters=splitters,
                                    last_valid_actual=date,
                                    exo_filler="mean",
                                    parallel_batch=self.tuner.parallel_batch,
                                )
                            }
                            modelling_tasks |= {
                                "Agg Model Strategy": AggModelStrategyTask(
                                    forecast_dir=self.tuner.result_dir,
                                    result_dir=self.tuner.selection_dir,
                                    aggregation_function=self.tuner.strategies[
                                        list(self.tuner.strategies.keys())[strategy]
                                    ],
                                    mode="recreate",
                                )
                            }
                            pipelines = OrderedDict()
                            pipelines |= {
                                "Modelling Pipeline": TaskPipeline(
                                    tasks=modelling_tasks
                                ),
                            }
                            process_pipeline = ProcessPipeline(pipelines=pipelines)
                            process_pipeline.run()
                            ev = Evaluator(
                                selected_data_dir=self.tuner.selection_dir,
                                reference_data_dir=self.tuner.input_dir,
                                metrics=list(ev_metrics.values()),
                            )
                            results = ev.evaluate()
                            results["execution_time"] = (
                                process_pipeline.execution_time.seconds
                            )
                            results["compared"] = -ev.compared
                            sub_F.append([*results.values()])
                            sub_G.append(
                                results["execution_time"] - time_constraint * 3600
                            )
                    F.append(np.mean(sub_F, axis=0))
                    G.append(np.mean(sub_G, axis=0))
                    to_pickle = {}
                    for key, value in zip(list(results.keys()), np.mean(sub_F, axis=0)):
                        to_pickle[key] = value
                    to_pickle["SOLUTION"] = vec
                    with open(
                        os.path.join(
                            result_dir, f"result_{self.counter}_{iteratation_}.pickle"
                        ),
                        "wb",
                    ) as f:
                        pickle.dump(to_pickle, f)
                out["F"] = F
                out["G"] = G

        return TunerProblem(self, ev_metrics)
