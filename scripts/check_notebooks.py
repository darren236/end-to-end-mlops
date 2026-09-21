"""Execute every project notebook without modifying its committed outputs."""

from pathlib import Path

import nbformat
from nbclient import NotebookClient

PROJECT_ROOT = Path(__file__).resolve().parents[1]
NOTEBOOK_DIR = PROJECT_ROOT / "notebooks"


def main() -> None:
    notebook_paths = sorted(NOTEBOOK_DIR.glob("*.ipynb"))
    if not notebook_paths:
        raise RuntimeError(f"No notebooks found in {NOTEBOOK_DIR}")

    for path in notebook_paths:
        notebook = nbformat.read(path, as_version=4)
        client = NotebookClient(
            notebook,
            timeout=180,
            kernel_name="python3",
            resources={"metadata": {"path": str(PROJECT_ROOT)}},
        )
        client.execute()
        print(f"Executed {path.relative_to(PROJECT_ROOT)}")


if __name__ == "__main__":
    main()
