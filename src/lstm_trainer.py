try:
    from .training_module.cli import lstm_main
except ImportError:
    from training_module.cli import lstm_main


main = lstm_main


if __name__ == "__main__":
    lstm_main()
