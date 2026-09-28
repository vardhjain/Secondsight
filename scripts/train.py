"""Repository shim for the training CLI; the implementation lives in :mod:`reid.cli.train`."""

from reid.cli.train import main

if __name__ == "__main__":
    raise SystemExit(main())
