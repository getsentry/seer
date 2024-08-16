import enum


class LoadingResult(enum.IntEnum):
    FAILED = -1
    PENDING = 0
    LOADING = 1
    DONE = 2
