try:
    from .training_module.cli import training_pipeline_main
except ImportError:
    from training_module.cli import training_pipeline_main


main = training_pipeline_main


if __name__ == "__main__":
    training_pipeline_main()
