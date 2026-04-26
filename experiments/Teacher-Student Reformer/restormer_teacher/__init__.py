"""Restormer-style teacher for same-resolution 256x256 restoration (Lab 3 distillation path)."""

from .config import TEACHER_MODEL_VERSION
from .model import RestormerTeacher, build_teacher_model

__all__ = ["RestormerTeacher", "build_teacher_model", "TEACHER_MODEL_VERSION"]
