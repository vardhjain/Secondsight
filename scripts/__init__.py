"""Repository shims that run the Secondsight CLIs from a source checkout.

The implementations live in :mod:`reid.cli` and are installed as the
``reid-*`` console scripts. These modules only forward to them so that
``python -m scripts.<name>`` keeps working in a clone.
"""
