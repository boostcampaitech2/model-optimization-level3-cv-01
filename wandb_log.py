import optuna
from typing import Any, Dict
import sys
from typing import Any, Dict, Sequence, Optional, Union
import wandb

class WeightsAndBiasesCallback(object):

    def __init__(
        self,
        metric_name: Union[str, Sequence[str]] = "value",
        wandb_kwargs: Optional[Dict[str, Any]] = None,
    ) -> None:

        if not isinstance(metric_name, Sequence):
            raise TypeError(
                "Expected metric_name to be string or sequence of strings, got {}.".format(
                    type(metric_name)
                )
            )

        self._metric_name = metric_name
        self._wandb_kwargs = wandb_kwargs or {}
        self.data = []

    def __call__(self, study: optuna.study.Study, trial: optuna.trial.FrozenTrial) -> None:

        run = wandb.init(**self._wandb_kwargs)

        if isinstance(self._metric_name, str):
            if len(trial.values) > 1:
                # Broadcast default name for multi-objective optimization.
                names = ["{}_{}".format(self._metric_name, i) for i in range(len(trial.values))]

            else:
                names = [self._metric_name]

        else:
            if len(self._metric_name) != len(trial.values):
                raise ValueError(
                    "Running multi-objective optimization "
                    "with {} objective values, but {} names specified. "
                    "Match objective values and names, or use default broadcasting.".format(
                        len(trial.values), len(self._metric_name)
                    )
                )

            else:
                names = [*self._metric_name]

        metrics = {name: value for name, value in zip(names, trial.values)}
        attributes = {"direction": [d.name for d in study.directions]}

        wandb.config.update(attributes)
        wandb.log({**trial.params, **metrics}, step=trial.number)
        run.finish
        # temp_values = trial.values
        # temp_names = names
        # temp_values.append(trial.number)
        # temp_names.append(str("step"))
        # self.data.append(temp_values)
        # table = wandb.Table(data=self.data, columns = temp_names)
        # wandb.log({"scatter_plot" : wandb.plot.scatter(table, "f1_score", "mean_time",
        #                          title="Custom Y vs X Scatter Plot")})

    def _initialize_run(self) -> None:
        """Initializes Weights & Biases run."""

        wandb.init(**self._wandb_kwargs)