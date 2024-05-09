from enum import StrEnum


class CeleryQueues(StrEnum):
    DEFAULT = "seer"
    CUDA = "seer-cuda"
