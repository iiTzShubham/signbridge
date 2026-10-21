"""Train and test the temporal Vision Transformer using the shared protocol."""
import sys
try:
    from .compare_models import main as _main
except ImportError:
    from training.compare_models import main as _main


if __name__ == "__main__":
    sys.argv.extend(["--models", "vit_temporal"])
    raise SystemExit(_main())
