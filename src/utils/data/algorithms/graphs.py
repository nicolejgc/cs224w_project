import clrs
import numpy as np
import chex
import networkx as nx

from .specs import SPECS
from clrs._src import probing
from clrs._src.probing import ProbesDict
from typing import Tuple


_Stage = clrs.Stage
_Location = clrs.Location
_Type = clrs.Type
_Array = np.ndarray
_Out = Tuple[_Array, ProbesDict]
_OutputClass = clrs.OutputClass


def _ff_impl(A: _Array, s: int, t: int, probes, w):
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
            'mask': np.copy(msk),
            'd': np.copy(d),
            'pi_h': np.copy(pi),
            'f_h': np.copy(f),
            'df': np.copy(df),
            'c_h': np.copy(C),
            '__is_bfs_op': np.copy([1])
        })

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
                    'pi_h': np.copy(pi),
                    'd': np.copy(prev_d),
                    'mask': np.copy(msk),
                    'f_h': np.copy(f),
                    'df': np.copy(df),
                    'c_h': np.copy(C),
                    '__is_bfs_op': np.copy([1])
                })

            if np.all(d == prev_d):
                break

        if pi[t] == t:
            break

        df = min([
            A[u, v] - f[u, v]
            for u, v in reverse(pi)
        ])

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
                'pi_h': np.copy(pi),
                'd': np.copy(d),
                'mask': np.copy(msk),
                'f_h': np.copy(f),
                'df': np.copy(df),
                'c_h': np.copy(C),
                '__is_bfs_op': np.array([0])
            })

    return f, probes


def ford_fulkerson(A: _Array, s: int, t: int):

    chex.assert_rank(A, 2)
    probes = probing.initialize(SPECS['ford_fulkerson'])
    A_pos = np.arange(A.shape[0])

    rng = np.random.default_rng(0)

    w = rng.random(size=A.shape)
    w = np.maximum(w, w.T) * probing.graph(np.copy(A))

    probing.push(
        probes,
        _Stage.INPUT,
        next_probe={
            'pos': np.copy(A_pos) * 1.0 / A.shape[0],
            's': probing.mask_one(s, A.shape[0]),
            't': probing.mask_one(t, A.shape[0]),
            'A': np.copy(A),
            'adj': probing.graph(np.copy(A)),
            'w': np.copy(w),
        })

    f, probes = _ff_impl(A, s, t, probes, w)

    probing.push(
        probes,
        _Stage.OUTPUT,
        next_probe={
            'f': np.copy(f)
        }
    )
    probing.finalize(probes)

    return f, probes

def _minimum_cut(A, s, t):
    C = np.zeros((A.shape[0], 2))

    graph = nx.from_numpy_array(A)
    nx.set_edge_attributes(graph, {(i, j): A[i, j] for i, j in zip(*A.nonzero())},
                           name='capacity')

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
