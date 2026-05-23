import csv
from datetime import datetime
from pathlib import Path
from typing import Any


class CSVLogger:

    def __init__(self, log_dir: str | Path, run_name: str | None = None):
        self.log_dir = Path(log_dir)
        self.log_dir.mkdir(parents=True, exist_ok=True)
        if run_name is None:
            run_name = datetime.now().strftime("%Y%m%d_%H%M%S")
        self.run_name = run_name
        self.metrics_file = self.log_dir / f"{run_name}_metrics.csv"
        self._fields: list[str] | None = None

    def log(self, **kwargs: Any) -> None:
        if self._fields is None:
            self._fields = list(kwargs.keys())
            with open(self.metrics_file, "w", newline="") as f:
                writer = csv.DictWriter(f, fieldnames=self._fields)
                writer.writeheader()
        with open(self.metrics_file, "a", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=self._fields)
            writer.writerow({k: kwargs.get(k, "") for k in self._fields})
