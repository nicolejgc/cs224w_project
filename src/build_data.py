import os
from functools import partial
from math import log
from pathlib import Path

import networkx as nx
import numpy as np
import typer
from clrs._src.probing import ProbesDict
from numpy.random import default_rng
from numpy.typing import NDArray

from config.data import DATA_SETTINGS
from utils.data import algorithms
from utils.data.graphs import bipartite, erdos_renyi_full, two_community
from utils.io import dump
from utils.types import Algorithm

app = typer.Typer(pretty_exceptions_show_locals=False)


def max_flow_init(adj, rng, **kwargs):
    num_nodes = adj.shape[0]

    if kwargs["random_st"]:
        source = rng.choice(num_nodes // 2)
        target = rng.choice(range(num_nodes // 2 + 1, num_nodes))

        if source == target:
            target = (source + 1) % num_nodes
    else:
        source, target = 0, num_nodes - 1

    if kwargs["capacity"]:
        high = 10
        capacity: NDArray = (
            rng.integers(low=1, high=high, size=(num_nodes, num_nodes)) / high
        )
    else:
        capacity: NDArray = np.ones((num_nodes, num_nodes))

    capacity = np.maximum(capacity, capacity.T) * adj
    capacity = capacity * np.abs((np.eye(num_nodes) - 1))

    return capacity, source, target


_INITS = {
    Algorithm.ff: max_flow_init,
    Algorithm.ffmc: max_flow_init,
    Algorithm.prp: max_flow_init,
    Algorithm.prmc: max_flow_init,
}

_GRAPH_DISTRIB = {
    "two_community": two_community,
    "erdos_renyi": erdos_renyi_full,
    "bipartite": bipartite,
}


def sample_trajectory(alg, params) -> ProbesDict:
    algorithm_fn = getattr(algorithms, alg)
    f, probes = algorithm_fn(*params)
    validate_max_flow(f, *params)

    return probes


def validate_max_flow(f: np.ndarray, A: np.ndarray, s: int, t: int):
    """
    Validates the output of a maxflow alg

    Only raises error if a hard constraint is violated
    """
    n = A.shape[0]

    # skew symmetry aka f(u, v) == -f(v, u)
    if not np.allclose(f, -f.T):
        diff = np.abs(f + f.T)
        max_diff = np.max(diff)
        raise AssertionError(f"Skew symmetry violated. Max deviation: {max_diff}")

    # capacity constraint aka f(u, v) <= c(u, v)
    flow_violation = f > A + 1e-5
    if np.any(flow_violation):
        indices = np.argwhere(flow_violation)
        u, v = indices[0]
        raise AssertionError(
            f"Capacity violated at ({u}, {v}). Flow: {f[u, v]}, Cap: {A[u, v]}"
        )

    # flow conservation: flow_in = flow_out excluding s, t
    net_flow = np.sum(f, axis=1)
    internal_nodes = np.delete(np.arange(n), [s, t])
    if not np.allclose(net_flow[internal_nodes], 0, atol=1e-5):
        u = internal_nodes[np.argmax(np.abs(net_flow[internal_nodes]))]
        raise AssertionError(
            f"Flow conservation violated at node {u}. Net flow: {net_flow[u]}"
        )

    # final maxflow check vs networkx
    alg_flow_value = np.sum(f[s, :])

    G = nx.from_numpy_array(A, create_using=nx.DiGraph)
    nx_flow_value = nx.maximum_flow_value(G, s, t, capacity="weight")

    if not np.isclose(alg_flow_value, nx_flow_value, atol=1e-4):
        print(
            f"OPTIMALITY FAIL: Your flow = {alg_flow_value}, True max flow = {nx_flow_value}"
        )

    return


def main(
    alg: Algorithm,
    dataset_name: str = "default",
    graph_density: float = 0.35,
    outer_prob: float = 0.05,
    save_path: Path = Path("./data/clrs"),
    graph_distrib: str = "two_community",
    weighted: bool = False,
    directed: bool = False,
    seed: int | None = None,
):
    if seed is None:
        seed = int.from_bytes(os.urandom(2), byteorder="big")

    assert graph_distrib in ["two_community", "erdos_renyi", "bipartite"]

    distrib = _GRAPH_DISTRIB[graph_distrib]

    rng = default_rng(seed)
    init_fn = _INITS[alg]

    save_path = save_path / alg.value / dataset_name

    probs = {}
    graphs = {}
    extras = {}

    # First sample graphs (aids reproducibility).
    for split in DATA_SETTINGS.keys():
        num_nodes = DATA_SETTINGS[split]["length"]
        probs[split] = max(graph_density, 1.25 * log(num_nodes) / num_nodes)

        distrib = partial(distrib, outer_prob=outer_prob)
        graphs[split] = []
        for _ in range(DATA_SETTINGS[split]["num_samples"]):
            adj = distrib(
                num_nodes=num_nodes,
                prob=probs[split]
                if graph_distrib != "bipartite"
                else rng.uniform(low=graph_density, high=1),
                directed=directed,
                weighted=weighted,
                rng=rng,
            )
            graphs[split].append(adj)

    # Then run the algorithm for each of them.
    for split in DATA_SETTINGS.keys():
        extras[split] = dict()
        data: list[ProbesDict] = []
        num_nodes = DATA_SETTINGS[split]["length"]
        with typer.progressbar(
            range(DATA_SETTINGS[split]["num_samples"]), label=split
        ) as progress:
            for i in progress:
                params = init_fn(
                    graphs[split][i],
                    rng,
                    random_st=graph_distrib != "bipartite",
                    capacity=graph_distrib != "bipartite",
                )
                trajectory = sample_trajectory(alg, params)
                data.append(trajectory)

        key = list(data[0]["hint"]["node"].keys())[0]
        avg_length = []
        for d in data:
            avg_length.append(d["hint"]["node"][key]["data"].shape[0])

        # David's temp debugging prints
        # print(data[0].keys())
        # print(data[0]["input"].keys())
        # print(data[0]["input"]["node"].keys())
        # print(data[0]["input"]["edge"].keys())
        # print(data[0]["input"]["graph"].keys())
        # print(f"what is {key}")

        # print statistics
        extras[split]["max"] = max(avg_length)
        extras[split]["avg"] = sum(avg_length) / len(avg_length)
        print("[avg] traj len:", extras[split]["avg"])
        print("[max] traj len:", extras[split]["max"])

        dump(data, save_path / f"{split}_{alg.value}.pkl")

    dump(dict(seed=seed, graph_density=probs, **extras), save_path / "config.json")


if __name__ == "__main__":
    app.command()(main)
    app()
