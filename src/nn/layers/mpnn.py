from typing import Callable

import torch
from torch.nn import Linear, Module, Sequential
from torch.nn import functional as F

Inf = 1e6


class MpnnConv(Module):
    def __init__(
        self,
        in_channels: int,
        edge_channels: int,
        mid_channels: int,
        out_channels: int,
        net: Sequential,
        aggregator: str,
        mid_activation: Callable | None = None,
        activation: Callable | None = None,
        bias: bool = True,
    ):
        super(MpnnConv, self).__init__()

        self.in_channels = in_channels  # type: ignore
        self.out_channels = out_channels  # type: ignore

        self.m_1 = Linear(in_features=in_channels, out_features=mid_channels, bias=bias)
        self.m_2 = Linear(in_features=in_channels, out_features=mid_channels, bias=bias)
        self.m_e = Linear(
            in_features=edge_channels, out_features=mid_channels, bias=bias
        )

        self.o1 = Linear(in_features=in_channels, out_features=out_channels, bias=bias)
        self.o2 = Linear(in_features=mid_channels, out_features=out_channels, bias=bias)

        self.net = net
        self.mid_activation = mid_activation  # type: ignore
        self.activation = activation  # type: ignore

        if aggregator == "max":
            reduce = torch.amax
        elif aggregator == "sum":
            reduce = torch.sum
        elif aggregator == "mean":
            reduce = torch.mean
        else:
            raise NotImplementedError("Invalid type of aggregator function.")

        self.aggregator = aggregator  # type: ignore
        self.reduce = reduce  # type: ignore

        self.reset_parameters()

    def reset_parameters(self):
        self.m_1.reset_parameters()
        self.m_2.reset_parameters()
        self.m_e.reset_parameters()
        self.o1.reset_parameters()
        self.o2.reset_parameters()

    def forward(self, x, adj, edge_attr):
        """
        x : Tensor
            node feature matrix (batch_size x num_nodes x num_nodes_features)
        adj : Tensor
            adjacency matrix (batch_size x num_nodes x num_nodes)
        edge_attr : Tensor
            edge attributes (batch_size x num_nodes x num_nodes x num_edge_features)
        """

        batch_size, num_nodes, num_features = x.shape
        _, _, _, num_edge_features = edge_attr.shape

        msg_1 = self.m_1(x)
        msg_2 = self.m_2(x)
        msg_e = self.m_e(edge_attr)

        msg = msg_1.unsqueeze(1) + msg_2.unsqueeze(2) + msg_e
        if self.net is not None:
            msg = self.net(F.relu(msg))

        if self.mid_activation is not None:
            msg = self.mid_activation(msg)

        if self.aggregator == "mean":
            msg = (msg * adj.unsqueeze(-1)).sum(1)
            msg = msg / torch.sum(adj, dim=-1, keepdim=True)
        elif self.aggregator == "max":
            max_arg = torch.where(
                adj.unsqueeze(-1).bool(), msg, torch.tensor(-Inf).to(msg.device)
            )
            msg = self.reduce(max_arg, dim=1)
        else:
            msg = self.reduce(msg * adj.unsqueeze(-1), dim=1)

        h_1 = self.o1(x)
        h_2 = self.o2(msg)

        out = h_1 + h_2

        if self.activation is not None:
            out = self.activation(out)

        return out
