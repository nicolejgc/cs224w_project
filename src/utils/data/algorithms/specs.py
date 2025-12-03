import clrs
import types

_Stage = clrs.Stage
_Location = clrs.Location
_Type = clrs.Type

SPECS = types.MappingProxyType(
    {
        **clrs._src.specs.SPECS,
        "ford_fulkerson": {
            "pos": (_Stage.INPUT, _Location.NODE, _Type.SCALAR),
            "s": (_Stage.INPUT, _Location.NODE, _Type.MASK_ONE),
            "t": (_Stage.INPUT, _Location.NODE, _Type.MASK_ONE),
            "A": (_Stage.INPUT, _Location.EDGE, _Type.SCALAR),
            "adj": (_Stage.INPUT, _Location.EDGE, _Type.MASK),
            "w": (_Stage.INPUT, _Location.EDGE, _Type.SCALAR),
            "mask": (_Stage.HINT, _Location.NODE, _Type.MASK),
            "pi_h": (_Stage.HINT, _Location.NODE, _Type.POINTER),
            "__is_bfs_op": (_Stage.HINT, _Location.GRAPH, _Type.MASK),
            "f_h": (_Stage.HINT, _Location.EDGE, _Type.SCALAR),
            "f": (_Stage.OUTPUT, _Location.EDGE, _Type.SCALAR),
        },
        "ford_fulkerson_mincut": {
            "pos": (_Stage.INPUT, _Location.NODE, _Type.SCALAR),
            "s": (_Stage.INPUT, _Location.NODE, _Type.MASK_ONE),
            "t": (_Stage.INPUT, _Location.NODE, _Type.MASK_ONE),
            "A": (_Stage.INPUT, _Location.EDGE, _Type.SCALAR),
            "adj": (_Stage.INPUT, _Location.EDGE, _Type.MASK),
            "w": (_Stage.INPUT, _Location.EDGE, _Type.SCALAR),
            "mask": (_Stage.HINT, _Location.NODE, _Type.MASK),
            "pi_h": (_Stage.HINT, _Location.NODE, _Type.POINTER),
            "f_h": (_Stage.HINT, _Location.EDGE, _Type.SCALAR),
            "c_h": (_Stage.HINT, _Location.NODE, _Type.CATEGORICAL),
            "__is_bfs_op": (_Stage.HINT, _Location.GRAPH, _Type.MASK),
            "f": (_Stage.OUTPUT, _Location.EDGE, _Type.SCALAR),
            "c": (_Stage.OUTPUT, _Location.NODE, _Type.CATEGORICAL),
        },
        "push_relabel": {
            # inputs
            "pos": (_Stage.INPUT, _Location.NODE, _Type.SCALAR),
            "s": (_Stage.INPUT, _Location.NODE, _Type.MASK_ONE),
            "t": (_Stage.INPUT, _Location.NODE, _Type.MASK_ONE),
            "A": (_Stage.INPUT, _Location.EDGE, _Type.SCALAR),  # capacity
            "adj": (_Stage.INPUT, _Location.EDGE, _Type.MASK),  # adj matrix
            # hints
            "h": (_Stage.HINT, _Location.NODE, _Type.SCALAR),  # height labels
            "e": (_Stage.HINT, _Location.NODE, _Type.SCALAR),  # excess
            "f_h": (_Stage.HINT, _Location.EDGE, _Type.SCALAR),  # intermed flow assn
            "active_nodes": (_Stage.HINT, _Location.NODE, _Type.MASK),
            "phase": (clrs.Stage.HINT, clrs.Location.GRAPH, clrs.Type.MASK_ONE),
            # outputs
            "f": (_Stage.OUTPUT, _Location.EDGE, _Type.SCALAR),  # final flow assn
        },
        "push_relabel_mincut": {
            # inputs
            "pos": (_Stage.INPUT, _Location.NODE, _Type.SCALAR),
            "s": (_Stage.INPUT, _Location.NODE, _Type.MASK_ONE),
            "t": (_Stage.INPUT, _Location.NODE, _Type.MASK_ONE),
            "A": (_Stage.INPUT, _Location.EDGE, _Type.SCALAR),  # capacity
            "adj": (_Stage.INPUT, _Location.EDGE, _Type.MASK),  # adj matrix
            # hints
            "h": (_Stage.HINT, _Location.NODE, _Type.SCALAR),  # height labels
            "e": (_Stage.HINT, _Location.NODE, _Type.SCALAR),  # excess
            "f_h": (_Stage.HINT, _Location.EDGE, _Type.SCALAR),  # intermed flow assn
            "c_h": (_Stage.HINT, _Location.NODE, _Type.CATEGORICAL),  # cut hints
            "active_nodes": (_Stage.HINT, _Location.NODE, _Type.MASK),
            "phase": (clrs.Stage.HINT, clrs.Location.GRAPH, clrs.Type.MASK_ONE),
            # outputs
            "f": (_Stage.OUTPUT, _Location.EDGE, _Type.SCALAR),  # final flow assn
            "c": (_Stage.OUTPUT, _Location.NODE, _Type.CATEGORICAL),  # min-cut partition
        },
    }
)

ALGS = [
    *clrs._src.specs.CLRS_30_ALGS,
    "ford_fulkerson",
    "ford_fulkerson_mincut",
    "push_relabel",
    "push_relabel_mincut",
]
