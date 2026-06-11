"""Package version lookup for CLI ``--version`` flags."""

from __future__ import annotations


def package_version() -> str:
    """The installed ``vesp`` distribution version, or a source-tree sentinel.

    Resolved via ``importlib.metadata`` so the CLIs report the version of whatever is actually
    installed (wheel, editable install). Running straight from an uninstalled source tree has
    no distribution metadata -- return a recognizable sentinel instead of guessing.
    """

    try:
        from importlib.metadata import version

        return version("vesp")
    except Exception:
        return "0.0.0+source"
