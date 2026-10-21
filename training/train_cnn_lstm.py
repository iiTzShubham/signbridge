"""Train and test the CNN-LSTM using the shared experiment protocol."""
import sys
try:
    from .compare_models import main as _main
except ImportError:
    from training.compare_models import main as _main


if __name__ == "__main__":
    sys.argv.extend(["--models", "cnn_lstm"])
    raise SystemExit(_main())
