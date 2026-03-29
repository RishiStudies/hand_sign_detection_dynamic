"""DEPRECATED: Thin wrapper for backwards compatibility.
Use: python -m src.training_module.cli train-lstm <args> instead.
Or directly: from src.training_module.cli import lstm_main
"""

try:
    from .training_module.cli import lstm_main
except ImportError:
    from training_module.cli import lstm_main


main = lstm_main


if __name__ == "__main__":
    lstm_main()
