from enum import Enum


class Algorithm(str, Enum):
    ff = "ford_fulkerson"
    ffmc = "ford_fulkerson_mincut"
    prp = "push_relabel"
