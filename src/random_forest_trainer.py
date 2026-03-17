try:
    from .training_module.cli import random_forest_main
except ImportError:
    from training_module.cli import random_forest_main


main = random_forest_main


if __name__ == "__main__":
    random_forest_main()
