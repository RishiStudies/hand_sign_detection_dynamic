from importlib import import_module
from typing import Any

__all__ = ["TrainingService", "JOB_HANDLERS"]


def __getattr__(name: str) -> Any:
	if name == "TrainingService":
		return import_module(".service", __name__).TrainingService
	if name == "JOB_HANDLERS":
		return import_module(".jobs", __name__).JOB_HANDLERS
	raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
