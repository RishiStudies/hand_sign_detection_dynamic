import argparse
import json

from .service import TrainingService


def _add_profile_args(parser: argparse.ArgumentParser, default_profile: str = "full") -> None:
    parser.add_argument("--profile", choices=["pi_zero", "full"], default=default_profile)


def training_pipeline_main() -> None:
    parser = argparse.ArgumentParser(
        description="Training CLI wrapper for the canonical training_module package"
    )
    parser.add_argument(
        "--command",
        choices=["legacy", "preprocess", "train-rf", "evaluate", "package", "device-all"],
        default="legacy",
    )
    _add_profile_args(parser, default_profile="pi_zero")
    parser.add_argument("--max-classes", type=int, default=None)
    parser.add_argument("--max-videos-per-class", type=int, default=None)
    parser.add_argument("--sequence-length", type=int, default=None)
    parser.add_argument("--frame-stride", type=int, default=None)
    parser.add_argument("--json-file", default=None)
    parser.add_argument("--video-folder", default=None)
    parser.add_argument("--csv-path", default=None)
    parser.add_argument("--note", default="")
    parser.add_argument("--model", choices=["random_forest", "lstm", "all"], default="all")
    parser.add_argument("--data", choices=["csv", "wlasl"], default=None)
    parser.add_argument("--no-save", action="store_true")
    parser.add_argument("--low-end", action="store_true")

    args = parser.parse_args()
    trainer = TrainingService()
    profile = trainer.get_profile(args.profile)
    save_model = not args.no_save

    max_classes = args.max_classes if args.max_classes is not None else int(profile["max_classes"])
    max_videos = (
        args.max_videos_per_class
        if args.max_videos_per_class is not None
        else int(profile["max_videos_per_class"])
    )
    sequence_length = (
        args.sequence_length if args.sequence_length is not None else int(profile["sequence_length"])
    )
    frame_stride = args.frame_stride if args.frame_stride is not None else int(profile["frame_stride"])

    if args.command == "preprocess":
        trainer.process_wlasl_videos(
            json_file=args.json_file,
            video_folder=args.video_folder,
            save_data=save_model,
            max_classes=max_classes,
            max_videos_per_class=max_videos,
            sequence_length=sequence_length,
            frame_stride=frame_stride,
        )
        print(json.dumps(trainer.last_preprocess_summary, indent=2))
        return

    if args.command == "train-rf":
        metrics = trainer.train_random_forest_from_csv(
            data_path=args.csv_path,
            save_model=save_model,
            low_end=args.low_end,
            profile_name=args.profile,
        )
        print(json.dumps(metrics, indent=2))
        return

    if args.command == "evaluate":
        accuracy = trainer.evaluate_random_forest(data_path=args.csv_path)
        print(json.dumps({"accuracy": accuracy}, indent=2))
        return

    if args.command == "package":
        package_path = trainer.package_artifacts(profile_name=args.profile, note=args.note)
        print(json.dumps({"package_path": package_path}, indent=2))
        return

    if args.command == "device-all":
        result = trainer.run_device_pipeline(
            profile_name=args.profile,
            note=args.note,
            csv_path=args.csv_path,
            json_file=args.json_file,
            video_folder=args.video_folder,
            max_classes=max_classes,
            max_videos_per_class=max_videos,
            sequence_length=sequence_length,
            frame_stride=frame_stride,
        )
        print(json.dumps(result, indent=2))
        return

    if args.model in {"random_forest", "all"}:
        rf_metrics = trainer.train_random_forest_from_csv(
            data_path=args.csv_path,
            save_model=save_model,
            low_end=args.low_end,
            profile_name=args.profile,
        )
        print(json.dumps({"random_forest": rf_metrics}, indent=2))

    if args.model in {"lstm", "all"}:
        if args.data == "wlasl" or args.model == "all":
            x_values, y_values = trainer.process_wlasl_videos(
                json_file=args.json_file,
                video_folder=args.video_folder,
                save_data=save_model,
                max_classes=max_classes,
                max_videos_per_class=max_videos,
                sequence_length=sequence_length,
                frame_stride=frame_stride,
            )
            trainer.train_lstm(
                x_values=x_values,
                y_values=y_values,
                save_model=save_model,
                low_end=args.low_end or bool(profile.get("lstm_low_end", False)),
            )
        else:
            trainer.train_lstm(
                save_model=save_model,
                low_end=args.low_end or bool(profile.get("lstm_low_end", False)),
            )

        print(json.dumps({"lstm": trainer.last_metrics.get("lstm", {})}, indent=2))


def random_forest_main() -> None:
    parser = argparse.ArgumentParser(description="Random Forest training wrapper")
    parser.add_argument("--csv-path", default=None)
    _add_profile_args(parser)
    parser.add_argument("--low-end", action="store_true")
    parser.add_argument("--no-save", action="store_true")
    args = parser.parse_args()

    trainer = TrainingService()
    metrics = trainer.train_random_forest_from_csv(
        data_path=args.csv_path,
        save_model=not args.no_save,
        low_end=args.low_end,
        profile_name=args.profile,
        source="random_forest_trainer_wrapper",
    )
    print(json.dumps(metrics, indent=2))


def lstm_main() -> None:
    parser = argparse.ArgumentParser(description="LSTM training wrapper")
    parser.add_argument("--low-end", action="store_true")
    parser.add_argument("--no-save", action="store_true")
    args = parser.parse_args()

    trainer = TrainingService()
    trainer.train_lstm(
        save_model=not args.no_save,
        low_end=args.low_end,
        source="lstm_trainer_wrapper",
    )
    print(json.dumps(trainer.last_metrics.get("lstm", {}), indent=2))


def orchestrator_main() -> None:
    parser = argparse.ArgumentParser(description="Canonical training orchestrator wrapper")
    _add_profile_args(parser)
    parser.add_argument("--csv-path", default=None)
    parser.add_argument("--json-file", default=None)
    parser.add_argument("--video-folder", default=None)
    parser.add_argument("--max-classes", type=int, default=None)
    parser.add_argument("--max-videos-per-class", type=int, default=None)
    parser.add_argument("--sequence-length", type=int, default=None)
    parser.add_argument("--frame-stride", type=int, default=None)
    parser.add_argument("--note", default="")
    args = parser.parse_args()

    trainer = TrainingService()
    result = trainer.run_device_pipeline(
        profile_name=args.profile,
        note=args.note,
        csv_path=args.csv_path,
        json_file=args.json_file,
        video_folder=args.video_folder,
        max_classes=args.max_classes,
        max_videos_per_class=args.max_videos_per_class,
        sequence_length=args.sequence_length,
        frame_stride=args.frame_stride,
    )
    print(json.dumps(result, indent=2))