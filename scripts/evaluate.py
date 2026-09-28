"""Repository shim for ``reid-evaluate``; the implementation is :mod:`reid.cli.evaluate`."""

from reid.cli.evaluate import main

if __name__ == "__main__":
    raise SystemExit(main())
