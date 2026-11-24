"""Tarnkappe: fast, small detection of personal data in English and German text."""
from .detect import Detector  # noqa: F401
from .scan import detect_scan  # noqa: F401

__version__ = "0.5.0"
