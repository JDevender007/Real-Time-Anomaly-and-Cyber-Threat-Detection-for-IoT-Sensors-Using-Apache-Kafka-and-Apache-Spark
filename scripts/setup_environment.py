"""Setup environment."""
from __future__ import annotations

import sys
from pathlib import Path


def main():
    project_root = Path(__file__).resolve().parent.parent
    if str(project_root) not in sys.path:
        sys.path.insert(0, str(project_root))

    from src.config.settings import create_required_directories
    create_required_directories()
    print("Environment setup complete")


if __name__ == "__main__":
    main()
