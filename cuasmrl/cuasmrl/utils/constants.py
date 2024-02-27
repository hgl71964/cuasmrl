from enum import Enum, auto


class Status(Enum):
    SEGFAULT = auto()
    TESTFAIL = auto()
    OK = auto()
