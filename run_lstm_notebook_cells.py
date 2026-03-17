import json
from pathlib import Path

NOTEBOOK_PATH = Path("lstm_model_training.ipynb")

# Execute cells in dependency order from the existing notebook.
ORDERED_CELL_MARKERS = [
    "import tensorflow as tf",  # imports
    "cwd = Path.cwd()",  # load data
    "print('Sequence length:'",  # explore
    "y_cat = to_categorical",  # preprocess labels
    "X_train, X_val, y_train, y_val = train_test_split(",  # split
    "model = Sequential([",  # define model
    "model_obj.compile(",  # compile
    "history = model.fit(",  # train
    "val_loss, val_acc = model.evaluate",  # evaluate
    "model.save(lstm_model_path)",  # save artifacts
    "summary = {",  # summary
]


def get_code_cells(nb_json: dict) -> list[tuple[str, str]]:
    cells: list[tuple[str, str]] = []
    for cell in nb_json.get("cells", []):
        if cell.get("cell_type") != "code":
            continue
        cid = cell.get("id")
        if not cid:
            cid = "unknown"
        source = cell.get("source", [])
        if isinstance(source, list):
            code = "\n".join(source)
        else:
            code = str(source)
        cells.append((str(cid), code))
    return cells


def resolve_execution_order(code_cells: list[tuple[str, str]]) -> list[tuple[str, str]]:
    ordered: list[tuple[str, str]] = []
    missing: list[str] = []

    for marker in ORDERED_CELL_MARKERS:
        matched = next(((cid, code) for cid, code in code_cells if marker in code), None)
        if matched is None:
            missing.append(marker)
        else:
            ordered.append(matched)

    if missing:
        raise ValueError(f"Missing expected notebook code markers: {missing}")

    return ordered


def main() -> None:
    if not NOTEBOOK_PATH.exists():
        raise FileNotFoundError(f"Notebook not found: {NOTEBOOK_PATH}")

    nb = json.loads(NOTEBOOK_PATH.read_text(encoding="utf-8"))
    code_cells = get_code_cells(nb)
    ordered_cells = resolve_execution_order(code_cells)

    global_ns: dict[str, object] = {}

    for index, (cid, code) in enumerate(ordered_cells, start=1):
        print(f"\n--- Running step {index}/{len(ordered_cells)}: {cid} ---")
        exec(compile(code, f"{NOTEBOOK_PATH}:{cid}", "exec"), global_ns)

    summary = global_ns.get("summary")
    print("\nTraining run completed.")
    if summary is not None:
        print("Summary:")
        print(summary)


if __name__ == "__main__":
    main()
