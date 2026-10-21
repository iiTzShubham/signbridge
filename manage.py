#!/usr/bin/env python
"""Django's command-line utility for administrative tasks."""
import os
import sys


def main():
    if sys.version_info < (3, 10):
        version = ".".join(map(str, sys.version_info[:3]))
        raise SystemExit(
            f"SignBridge requires Python 3.10 or newer; this interpreter is {version}. "
            "Install Python 3.12, recreate .venv, and install requirements.txt."
        )
    os.environ.setdefault("DJANGO_SETTINGS_MODULE", "config.settings")
    try:
        from django.core.management import execute_from_command_line
    except ImportError as exc:
        raise ImportError(
            "Django is not installed. Install dependencies with: pip install -r requirements.txt"
        ) from exc
    execute_from_command_line(sys.argv)


if __name__ == "__main__":
    main()
