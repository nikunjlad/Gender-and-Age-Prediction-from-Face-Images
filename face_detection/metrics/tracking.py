"""Training-time metric tracking: running averages and optional MLflow logging.

MLflow is an experiment tracker. It stores each run's settings and metrics so runs can be
compared in its web UI. It is optional; set train.mlflow: true in the config to use it.
"""

from __future__ import annotations

from pathlib import Path


class Meter:
    """Keeps running averages of several values, weighted by batch size.

    Example:
        m = Meter()
        m.add({"loss": 2.0}, n=4)
        m.add({"loss": 1.0}, n=4)
        m.avg()  # {"loss": 1.5}
    """

    def __init__(self):
        self.sums, self.n = {}, 0

    def add(self, vals: dict, n: int = 1):
        """Adds one batch of values.

        Args:
            vals: Name -> value for this batch.
            n: Batch size, used as the weight.
        """
        for k, v in vals.items():
            self.sums[k] = self.sums.get(k, 0.0) + float(v) * n
        self.n += n

    def avg(self) -> dict:
        """Returns name -> weighted average of everything added so far."""
        return {k: v / max(self.n, 1) for k, v in self.sums.items()}


class MlflowLogger:
    """Thin MLflow wrapper that does nothing when disabled or when MLflow is missing.

    On creation it starts an MLflow run named after the run folder and records every
    config value as a parameter. Example: train.lr is logged as the parameter "train.lr".

    Args:
        enabled: Whether to log at all.
        cfg: The full config. cfg["name"] is used as the MLflow experiment name.
        run_dir: The run's output folder; its name becomes the MLflow run name.
    """

    def __init__(self, enabled: bool, cfg: dict, run_dir: Path):
        self.ml = None  # the mlflow module, or None when logging is off
        if not enabled:
            return
        try:
            import mlflow

            self.ml = mlflow
            mlflow.set_experiment(cfg["name"])
            mlflow.start_run(run_name=run_dir.name)
            flat = {}

            def walk(d, p=""):  # flatten nested config into dotted keys
                for k, v in d.items():
                    if isinstance(v, dict):
                        walk(v, p + k + ".")
                    else:
                        flat[p + k] = str(v)[:250]  # MLflow limits parameter value length

            walk(cfg)
            mlflow.log_params(flat)
        except Exception as e:  # pragma: no cover
            print(f"[mlflow] disabled: {e}")
            self.ml = None

    def log(self, metrics: dict, step: int):
        """Logs metrics for one step (epoch); None values are skipped."""
        if self.ml:
            self.ml.log_metrics({k: float(v) for k, v in metrics.items() if v is not None}, step=step)

    def artifact(self, path):
        """Uploads a file (for example a config or checkpoint) to the MLflow run."""
        if self.ml:
            self.ml.log_artifact(str(path))

    def close(self):
        """Ends the MLflow run."""
        if self.ml:
            self.ml.end_run()
