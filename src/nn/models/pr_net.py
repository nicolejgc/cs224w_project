"""
Push-Relabel Neural Network (PR_Net)
"""

from typing import Callable, Dict, List

import clrs
import torch

from nn import losses as loss
from nn.models.epd import EncodeProcessDecode_Impl as Net
from nn.models.impl import (
    _dimensions,
    _expand_to,
    _hints_i,
    decoders,
)
from utils import is_not_done_broadcast
from utils.data import adj_mat

Result = Dict[str, clrs.DataPoint]

_Feedback = clrs.Feedback
_Spec = clrs.Spec
_DataPoint = clrs.DataPoint


def _get_phase(hints, device, batch_size):
    """Extracts phase mask: 0.0 for Push-Relabel, 1.0 for Global BFS."""
    for hint in hints:
        if hint.name == "phase":
            phase = hint.data.to(device).float()
            if phase.dim() == 1:
                phase = phase.unsqueeze(-1)
            return phase

    return torch.zeros((batch_size, 1), device=device)


class PR_Net(clrs.Model):
    def __init__(
        self,
        spec: _Spec,
        num_hidden: int,
        optim_fn: Callable,
        dummy_trajectory: _Feedback,
        alpha: float,
        processor: str,
        aggregator: str,
        no_feats: List = ["adj"],
        add_noise: bool = False,
        decode_hints: bool = True,
        encode_hints: bool = True,
        max_steps: int | None = None,
    ):
        super().__init__(spec=spec)

        self.device = (
            "cuda"
            if torch.cuda.is_available()
            else ("mps" if torch.mps.is_available() else "cpu")
        )

        self.net_ = PRNet_Impl(
            spec=spec,
            dummy_trajectory=dummy_trajectory,
            processor=processor,
            aggregator=aggregator,
            num_hidden=num_hidden,
            encode_hints=encode_hints,
            decode_hints=decode_hints,
            max_steps=max_steps,
            no_feats=no_feats,
            add_noise=add_noise,
            device=self.device,
        )

        self.optimizer = optim_fn(self.net_.parameters())
        self.alpha = alpha
        self.no_feats = lambda x: x in no_feats or x.startswith("__")
        self.decode_hints = decode_hints
        self.spec = spec

    def dump_model(self, path):
        torch.save(self.net_.state_dict(), path)

    def restore_model(self, path, device):
        self.net_.load_state_dict(torch.load(path, map_location=device))

    def _train_step(self, feedback: _Feedback):
        self.net_.train()
        self.optimizer.zero_grad()

        preds, hint_preds = self.net_(feedback.features)
        total_loss = 0.0
        n_hints = 0

        if self.decode_hints:
            hint_loss = 0.0
            for truth in feedback.features.hints:
                if self.no_feats(truth.name):
                    continue
                n_hints += 1
                hint_loss += loss.hint_loss(
                    hint_preds, truth, feedback, self.alpha, self.device
                )

            if n_hints > 0:
                total_loss += hint_loss / n_hints

        for truth in feedback.outputs:
            total_loss += loss.output_loss(
                preds, truth, feedback, self.alpha, self.device
            )

        total_loss.backward()
        self.optimizer.step()

        return total_loss.item()

    def feedback(self, feedback: _Feedback) -> float:
        return self._train_step(feedback)

    @torch.no_grad()
    def predict(self, features: clrs.Features):
        self.net_.eval()
        raw_preds, aux = self.net_(features)
        preds = decoders.postprocess(raw_preds, self.spec)
        return preds, (raw_preds, aux)

    @torch.no_grad()
    def verbose_loss(self, feedback: _Feedback, preds, aux_preds):
        losses = {}
        total_loss = 0.0
        n_hints = 0

        for truth in feedback.features.hints:
            if self.no_feats(truth.name):
                continue
            n_hints += 1
            curr_loss = (
                loss.hint_loss(aux_preds, truth, feedback, self.alpha, self.device)
                .cpu()
                .item()
            )
            losses["aux_" + truth.name] = curr_loss
            total_loss += curr_loss

        if n_hints > 0:
            total_loss /= n_hints

        for truth in feedback.outputs:
            total_loss += loss.output_loss(
                preds, truth, feedback, self.alpha, self.device
            )

        return losses, total_loss


class PRNet_Impl(torch.nn.Module):
    def __init__(
        self,
        spec: _Spec,
        dummy_trajectory: _Feedback,
        num_hidden: int,
        encode_hints: bool,
        decode_hints: bool,
        processor: str,
        aggregator: str,
        no_feats: List,
        add_noise: bool = False,
        bias: bool = True,
        max_steps: int | None = None,
        device: str = "cpu",
    ):
        super().__init__()
        self.num_hidden = num_hidden
        self.decode_hints = decode_hints
        self.max_steps = max_steps
        self.encode_hints = encode_hints
        self.device = device
        self.spec = spec

        # Common args for sub-networks
        net_args = dict(
            spec=spec,
            dummy_trajectory=dummy_trajectory,
            num_hidden=num_hidden,
            encode_hints=encode_hints,
            decode_hints=decode_hints,
            processor=processor,
            aggregator=aggregator,
            no_feats=no_feats,
            add_noise=add_noise,
            device=device,
        )

        self.push_relabel_net = Net(**net_args)  # ty:ignore[invalid-argument-type]
        self.push_relabel_net.encoders.pop("c_h")

        self.global_relabel_net = Net(**net_args)  # ty:ignore[invalid-argument-type]
        for key in ["f_h", "c_h"]:
            self.global_relabel_net.encoders.pop(key)

        self.to(device)

    def op(self, trajectories, h_pr, adj, phase):
        """
        Executes both networks and blends results based on phase mask.
        """
        _, h_pr, h_preds_pr = self.push_relabel_net.step(trajectories, h_pr, adj)

        cand_bfs, _, h_preds_bfs = self.global_relabel_net.step(
            trajectories, h_pr.detach(), adj
        )

        if self.decode_hints:
            hint_preds = {}
            for name, pred_pr in h_preds_pr.items():
                pred_bfs = h_preds_bfs.get(name, torch.zeros_like(pred_pr))

                ndim = pred_pr.dim()
                bfs_mask_expanded = _expand_to(phase, ndim)
                pr_mask_expanded = 1.0 - bfs_mask_expanded

                hint_preds[name] = (pred_pr * pr_mask_expanded) + (
                    pred_bfs * bfs_mask_expanded
                )
        else:
            hint_preds = {}

        pr_mask_hidden = 1.0 - _expand_to(phase, h_pr.dim())
        h_pr = h_pr * pr_mask_hidden

        cand = {}
        for name, pred_bfs in cand_bfs.items():
            # Mask out outputs during PR phase (outputs usually only valid at end or specific steps)
            mask_cand = _expand_to(phase, pred_bfs.dim())
            cand[name] = pred_bfs * mask_cand

        return cand, h_pr, hint_preds

    def forward(self, features):
        output_preds = {}
        hint_preds = []

        num_steps = (
            self.max_steps if self.max_steps else features.hints[0].data.shape[0] - 1
        )
        batch_size, num_nodes = _dimensions(features.inputs)

        h_pr = torch.zeros((batch_size, num_nodes, self.num_hidden)).to(self.device)
        adj = adj_mat(features).to(self.device)

        for i in range(num_steps):
            trajectories = [features.inputs]
            if self.encode_hints:
                trajectories.append(_hints_i(features.hints, i))

            if self.decode_hints:
                phase = _get_phase(_hints_i(features.hints, i), self.device, batch_size)
            else:
                phase = torch.zeros((batch_size, 1)).to(self.device)

            cand, h_pr, h_preds = self.op(trajectories, h_pr, adj, phase)

            if "f" in cand:
                cand["f"] = cand["f"] * adj

            if "f_h" in h_preds:
                h_preds["f_h"] = h_preds["f_h"] * adj

            hint_preds.append(h_preds)

            for name in cand:
                if name not in output_preds or features.lengths.sum() == 0:
                    output_preds[name] = cand[name]
                else:
                    is_not_done = is_not_done_broadcast(features.lengths, i, cand[name])
                    output_preds[name] = (
                        is_not_done * cand[name]
                        + (1.0 - is_not_done) * output_preds[name]
                    )

        return output_preds, hint_preds
