"""Repository shim for ``reid-visualize``; the implementation is :mod:`reid.cli.visualize`."""

from reid.cli.visualize import main

if __name__ == "__main__":
    raise SystemExit(main())
