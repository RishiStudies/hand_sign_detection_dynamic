"""DEPRECATED: Thin wrapper for backwards compatibility.
Use: python -m src.training_module.cli train-rf <args> instead.
Or directly: from src.training_module.cli import random_forest_main
"""

try:
    from .training_module.cli import random_forest_main
except ImportError:
    from training_module.cli import random_forest_main


main = random_forest_main


if __name__ == "__main__":
    random_forest_main()
