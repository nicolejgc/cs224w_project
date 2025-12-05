from enum import Enum


class Algorithm(str, Enum):
    ff = "ford_fulkerson"
    ffmc = "ford_fulkerson_mincut"
    prp = "push_relabel"
    prmc = "push_relabel_mincut"
    # Transfer learning variants (vessel features)
    ffmc_vessel = "ford_fulkerson_mincut_vessel"
    prmc_vessel = "push_relabel_mincut_vessel"
