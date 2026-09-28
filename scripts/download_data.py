"""Repository shim for the ``reid-download`` CLI (see :mod:`reid.cli.download`)."""

from reid.cli.download import main

if __name__ == "__main__":
    raise SystemExit(main())
