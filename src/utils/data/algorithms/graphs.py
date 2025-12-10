import clrs
import numpy as np
import chex
import networkx as nx

from .specs import SPECS
from clrs._src import probing
from clrs._src.probing import ProbesDict
from enum import IntEnum
from typing import Tuple


_Stage = clrs.Stage
_Location = clrs.Location
_Type = clrs.Type
_Array = np.ndarray
_Out = Tuple[_Array, ProbesDict]
_OutputClass = clrs.OutputClass


def _ff_impl(A: _Array, s: int, t: int, probes, w) -> _Out:
    f = np.zeros((A.shape[0], A.shape[0]))
    df = np.array(0)

    C = _minimum_cut(A, s, t)

    def reverse(pi):
        u, v = pi[t], t
        while u != v:
            yield u, v
            v = u
            u = pi[u]

    d = np.zeros(A.shape[0])
    msk = np.zeros(A.shape[0])
    pi = np.arange(A.shape[0])
    d[s] = 0
    msk[s] = 1

    probing.push(
        probes,
        _Stage.HINT,
        next_probe={
            "mask": np.copy(msk),
            "d": np.copy(d),
            "pi_h": np.copy(pi),
            "f_h": np.copy(f),
            "df": np.copy(df),
            "c_h": np.copy(C),
            "__is_bfs_op": np.copy([1]),
        },
    )

    while True:
        for _ in range(A.shape[0]):
            prev_d = np.copy(d)
            prev_msk = np.copy(msk)
            for u in range(A.shape[0]):
                for v in range(A.shape[0]):
                    if prev_msk[u] == 1 and A[u, v] - abs(f[u, v]) > 0:
                        if msk[v] == 0 or prev_d[u] + w[u, v] < d[v]:
                            d[v] = prev_d[u] + w[u, v]
                            pi[v] = u
                        msk[v] = 1

            probing.push(
                probes,
                _Stage.HINT,
                next_probe={
                    "pi_h": np.copy(pi),
                    "d": np.copy(prev_d),
                    "mask": np.copy(msk),
                    "f_h": np.copy(f),
                    "df": np.copy(df),
                    "c_h": np.copy(C),
                    "__is_bfs_op": np.copy([1]),
                },
            )

            if np.all(d == prev_d):
                break

        if pi[t] == t:
            break

        df = min([A[u, v] - f[u, v] for u, v in reverse(pi)])

        for u, v in reverse(pi):
            f[u, v] += df
            f[v, u] -= df

        d = np.zeros(A.shape[0])
        msk = np.zeros(A.shape[0])
        pi = np.arange(A.shape[0])
        d[s] = 0
        msk[s] = 1
        probing.push(
            probes,
            _Stage.HINT,
            next_probe={
                "pi_h": np.copy(pi),
                "d": np.copy(d),
                "mask": np.copy(msk),
                "f_h": np.copy(f),
                "df": np.copy(df),
                "c_h": np.copy(C),
                "__is_bfs_op": np.array([0]),
            },
        )

    return f, probes


def ford_fulkerson(A: _Array, s: int, t: int) -> _Out:
    chex.assert_rank(A, 2)
    probes = probing.initialize(SPECS["ford_fulkerson"])
    A_pos = np.arange(A.shape[0])

    rng = np.random.default_rng(0)

    w = rng.random(size=A.shape)
    w = np.maximum(w, w.T) * probing.graph(np.copy(A))

    probing.push(
        probes,
        _Stage.INPUT,
        next_probe={
            "pos": np.copy(A_pos) * 1.0 / A.shape[0],
            "s": probing.mask_one(s, A.shape[0]),
            "t": probing.mask_one(t, A.shape[0]),
            "A": np.copy(A),
            "adj": probing.graph(np.copy(A)),
            "w": np.copy(w),
        },
    )

    f, probes = _ff_impl(A, s, t, probes, w)

    probing.push(probes, _Stage.OUTPUT, next_probe={"f": np.copy(f)})
    probing.finalize(probes)

    return f, probes


def ford_fulkerson_mincut(A: _Array, s: int, t: int) -> _Out:
    chex.assert_rank(A, 2)
    probes = probing.initialize(SPECS["ford_fulkerson_mincut"])
    A_pos = np.arange(A.shape[0])

    rng = np.random.default_rng(0)

    w = rng.random(size=A.shape)
    w = np.maximum(w, w.T) * probing.graph(np.copy(A))

    probing.push(
        probes,
        _Stage.INPUT,
        next_probe={
            "pos": np.copy(A_pos) * 1.0 / A.shape[0],
            "s": probing.mask_one(s, A.shape[0]),
            "t": probing.mask_one(t, A.shape[0]),
            "A": np.copy(A),
            "adj": probing.graph(np.copy(A)),
            "w": np.copy(w),
        },
    )

    f, probes = _ff_impl(A, s, t, probes, w)

    probing.push(
        probes, _Stage.OUTPUT, next_probe={"f": np.copy(f), "c": _minimum_cut(A, s, t)}
    )

    probing.finalize(probes)

    return f, probes


def ford_fulkerson_mincut_vessel(
    length: _Array, 
    distance: _Array, 
    curveness: _Array, 
    adj: _Array,
    s: int, 
    t: int
) -> _Out:
    """
    Ford-Fulkerson min-cut with vessel features instead of capacity.
    
    For transfer learning: the algorithm runs the same, but inputs are vessel
    features. The model learns to map vessel features -> effective capacity.
    
    Args:
        length: [N, N] vessel segment lengths
        distance: [N, N] Euclidean distances
        curveness: [N, N] vessel tortuosity (1.0 = straight)
        adj: [N, N] adjacency mask
        s: source node
        t: target node
    
    Returns:
        f: final flow assignment
        probes: algorithm trace with hints
    """
    chex.assert_rank(length, 2)
    n = length.shape[0]
    
    # Use uniform capacity = 1.0 for all edges
    # The MODEL learns to predict flow from vessel features
    # Ground truth is computed with uniform capacity to keep flow values bounded
    A = adj * 1.0  # All edges have capacity 1.0
    
    # Alternative: derive capacity from vessel features (normalized to [0,1])
    # Uncomment to use: straighter, shorter vessels have higher capacity
    # capacity_score = (1.0 - length) * curveness  # both in [0,1], higher = better
    # A = adj * (0.1 + 0.9 * capacity_score)  # Capacity in [0.1, 1.0]
    
    probes = probing.initialize(SPECS["ford_fulkerson_mincut_vessel"])
    A_pos = np.arange(n)
    
    rng = np.random.default_rng(0)
    w = rng.random(size=A.shape)
    w = np.maximum(w, w.T) * adj
    
    # Push INPUT with vessel features (not capacity!)
    probing.push(
        probes,
        _Stage.INPUT,
        next_probe={
            "pos": np.copy(A_pos) * 1.0 / n,
            "s": probing.mask_one(s, n),
            "t": probing.mask_one(t, n),
            "length": np.copy(length),
            "distance": np.copy(distance),
            "curveness": np.copy(curveness),
            "adj": np.copy(adj),
        },
    )
    
    # Run Ford-Fulkerson with derived capacity
    f, probes = _ff_impl(A, s, t, probes, w)
    
    # Output: flow and min-cut
    probing.push(
        probes, _Stage.OUTPUT, next_probe={"f": np.copy(f), "c": _minimum_cut(A, s, t)}
    )
    
    probing.finalize(probes)
    
    return f, probes


def _minimum_cut(A, s, t):
    C = np.zeros((A.shape[0], 2))

    graph = nx.from_numpy_array(A)
    nx.set_edge_attributes(
        graph, {(i, j): A[i, j] for i, j in zip(*A.nonzero())}, name="capacity"
    )

    _, cuts = nx.minimum_cut(graph, s, t)

    for v in cuts[0]:
        C[v][0] = 1

    for v in cuts[1]:
        C[v][1] = 1

    return C


def _masked_array(a):
    a = np.empty_like(a)
    a.fill(_OutputClass.MASKED)
    return a


class PushRelabelPhase(IntEnum):
    PUSH_RELABEL = 0
    BFS = 1


def run_global_relabel(probes, A, f, h, e, s, t, n, C=None):
    """
    Runs a backwards BFS on residual graph to relabel.

    Args:
        C: Optional min-cut array for push_relabel_mincut variant
    """
    d = np.full(n, n, dtype=int)
    d[t] = 0

    q = [t]
    visited = np.zeros(n, dtype=bool)
    visited[t] = True

    while len(q) > 0:
        current_layer = np.zeros(n)
        current_layer[q] = 1.0

        hint_probe = {
            "h": np.copy(d),
            "e": np.copy(e),
            "f_h": np.copy(f),
            "active_nodes": current_layer,
            "__phase": np.array([PushRelabelPhase.BFS]),
        }

        # Include c_h if we're doing mincut variant
        if C is not None:
            hint_probe["c_h"] = np.copy(C)

        probing.push(probes, clrs.Stage.HINT, next_probe=hint_probe)

        new_q = []
        for v in q:
            for u in range(n):
                # We can move from u to v if capacity(u,v) - flow(u,v) > 0
                # aka do BFS on residual graph.
                res_cap = A[u, v] - f[u, v]

                if res_cap > 0 and not visited[u]:
                    visited[u] = True
                    d[u] = d[v] + 1
                    new_q.append(u)
        q = new_q

    mask = d < n
    h[mask] = d[mask]
    h[s] = n

    return h


def _push_relabel_impl(A: np.ndarray, s: int, t: int, C: np.ndarray | None) -> _Out:
    """
    Push-relabel with global relabel heuristic and batching
    """
    n = A.shape[0]
    probes = probing.initialize(SPECS["push_relabel"])

    f = np.zeros_like(A)
    h = np.zeros(n, dtype=int)
    e = np.zeros(n)

    # Preflow: saturate source edges
    h[s] = n
    for v in range(n):
        if A[s, v] > 0:
            flow = A[s, v]
            f[s, v] = flow
            f[v, s] = -flow
            e[v] += flow
            e[s] -= flow

    probing.push(
        probes,
        clrs.Stage.INPUT,
        next_probe={
            "pos": np.arange(n) * 1.0 / n,
            "s": probing.mask_one(s, n),
            "t": probing.mask_one(t, n),
            "A": np.copy(A),
            "adj": (A > 0).astype(float),
        },
    )

    global_relabel_freq = n
    steps_since_relabel = 0
    step = 0

    while True:
        # perform global-relabel using BFS
        if steps_since_relabel >= global_relabel_freq or step == 0:
            # global relabel also adds probes for BFS steps
            h = run_global_relabel(probes, A, f, h, e, s, t, n)
            steps_since_relabel = 0
            step += 1

        # Get active nodes
        active_mask = e > 0
        active_mask[s] = False
        active_mask[t] = False
        current_active_nodes = np.where(active_mask)[0]

        # Check if we are done
        if len(current_active_nodes) == 0:
            # Necessary to avoid silly edge cases
            if step == 0:
                probing.push(
                    probes,
                    clrs.Stage.HINT,
                    next_probe={
                        "h": np.copy(h),
                        "e": np.copy(e),
                        "f_h": np.copy(f),
                        "active_nodes": np.zeros(n),
                        "__phase": np.array([PushRelabelPhase.PUSH_RELABEL]),
                    },
                )
            break

        probing.push(
            probes,
            clrs.Stage.HINT,
            next_probe={
                "h": np.copy(h),
                "e": np.copy(e),
                "f_h": np.copy(f),
                "active_nodes": active_mask.astype(float),
                "__phase": np.array([PushRelabelPhase.PUSH_RELABEL]),
            },
        )
        steps_since_relabel += 1
        step += 1

        # Discharge stage
        for u in current_active_nodes:
            if e[u] <= 0:
                continue

            # Pushing
            for v in range(n):
                residual = A[u, v] - f[u, v]
                if residual > 0 and h[u] == h[v] + 1:
                    delta = min(e[u], residual)
                    if delta > 0:
                        f[u, v] += delta
                        f[v, u] -= delta
                        e[u] -= delta
                        e[v] += delta
                        if e[u] == 0:
                            break

            # Relabeling
            if e[u] > 0:
                min_h = np.inf
                for v in range(n):
                    residual = A[u, v] - f[u, v]
                    if residual > 0:
                        min_h = min(min_h, h[v])

                if min_h != np.inf:
                    h[u] = min_h + 1

    probing.push(probes, clrs.Stage.OUTPUT, next_probe={"f": np.copy(f)})
    probing.finalize(probes)

    return f, probes


def push_relabel(A: np.ndarray, s: int, t: int) -> _Out:
    return _push_relabel_impl(A, s, t, None)


def push_relabel_mincut(A: np.ndarray, s: int, t: int) -> _Out:
    """Push-relabel algorithm with min-cut output."""
    n = A.shape[0]
    probes = probing.initialize(SPECS["push_relabel_mincut"])

    f = np.zeros_like(A)
    h = np.zeros(n, dtype=int)
    e = np.zeros(n)
    C = _minimum_cut(A, s, t)  # Get the min-cut partition

    # Preflow: saturate source edges
    h[s] = n
    for v in range(n):
        if A[s, v] > 0:
            flow = A[s, v]
            f[s, v] = flow
            f[v, s] = -flow
            e[v] += flow
            e[s] -= flow

    probing.push(
        probes,
        clrs.Stage.INPUT,
        next_probe={
            "pos": np.arange(n) * 1.0 / n,
            "s": probing.mask_one(s, n),
            "t": probing.mask_one(t, n),
            "A": np.copy(A),
            "adj": (A > 0).astype(float),
        },
    )

    global_relabel_freq = n
    steps_since_relabel = 0
    step = 0

    while True:
        # perform global-relabel using BFS
        if steps_since_relabel >= global_relabel_freq or step == 0:
            h = run_global_relabel(probes, A, f, h, e, s, t, n, C)
            steps_since_relabel = 0
            step += 1

        # Get active nodes
        active_mask = e > 0
        active_mask[s] = False
        active_mask[t] = False
        current_active_nodes = np.where(active_mask)[0]

        # Check if we are done
        if len(current_active_nodes) == 0:
            if step == 0:
                probing.push(
                    probes,
                    clrs.Stage.HINT,
                    next_probe={
                        "h": np.copy(h),
                        "e": np.copy(e),
                        "f_h": np.copy(f),
                        "c_h": np.copy(C),
                        "active_nodes": np.zeros(n),
                        "__phase": np.array([PushRelabelPhase.PUSH_RELABEL]),
                    },
                )
            break

        probing.push(
            probes,
            clrs.Stage.HINT,
            next_probe={
                "h": np.copy(h),
                "e": np.copy(e),
                "f_h": np.copy(f),
                "c_h": np.copy(C),
                "active_nodes": active_mask.astype(float),
                "__phase": np.array([PushRelabelPhase.PUSH_RELABEL]),
            },
        )
        steps_since_relabel += 1
        step += 1

        # Discharge stage
        for u in current_active_nodes:
            if e[u] <= 0:
                continue

            # Pushing
            for v in range(n):
                residual = A[u, v] - f[u, v]
                if residual > 0 and h[u] == h[v] + 1:
                    delta = min(e[u], residual)
                    if delta > 0:
                        f[u, v] += delta
                        f[v, u] -= delta
                        e[u] -= delta
                        e[v] += delta
                        if e[u] == 0:
                            break

            # Relabeling
            if e[u] > 0:
                min_h = np.inf
                for v in range(n):
                    residual = A[u, v] - f[u, v]
                    if residual > 0:
                        min_h = min(min_h, h[v])

                if min_h != np.inf:
                    h[u] = min_h + 1

    probing.push(
        probes,
        clrs.Stage.OUTPUT,
        next_probe={
            "f": np.copy(f),
            "c": np.copy(C),
        },
    )
    probing.finalize(probes)

    return f, probes
