from random import random
from typing import Callable, Dict, List

import clrs
import torch

from nn import losses as loss
from nn.models.epd import EncodeProcessDecode_Impl as Net
from nn.models.impl import (
    _dimensions,
    _expand_to,
    _hints_i,
    _own_hints_i,
    decoders,
)
from utils import is_not_done_broadcast
from utils.data import adj_mat, edge_attr_mat

Result = Dict[str, clrs.DataPoint]

_Feedback = clrs.Feedback
_Spec = clrs.Spec
_DataPoint = clrs.DataPoint


def _get_phase(hints, device, batch_size):
    """Extracts phase mask: 0.0 for Push-Relabel, 1.0 for Global BFS."""
    for hint in hints:
        if hint.name == "phase" or hint.name == "__phase":
            phase = hint.data.to(device).float()
            if phase.dim() == 1:
                phase = phase.unsqueeze(-1)
            return phase

    return torch.zeros((batch_size, 1), device=device)


def _match_phase(phase, target):
    if target.dim() == 1 and phase.dim() == 2 and phase.shape[1] == 1:
        return phase.squeeze(-1)
    return _expand_to(phase, target.dim())


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
        annealing: bool = True,
        device: str | None = None,
    ):
        super().__init__(spec=spec)

        if device is not None:
            self.device = device
        else:
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
            annealing=annealing,
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

        # denormalize height
        if "h" in preds:
            _, num_nodes = _dimensions(features.inputs)
            preds["h"].data = preds["h"].data * num_nodes

            if isinstance(aux, list):
                for step_preds in aux:
                    if "h" in step_preds:
                        step_preds["h"] = step_preds["h"] * num_nodes

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

        return losses, (
            total_loss.item() if isinstance(total_loss, torch.Tensor) else 0.0
        )


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
        annealing: bool = True,
        device: str = "cpu",
    ):
        super().__init__()
        self.num_hidden = num_hidden
        self.decode_hints = decode_hints
        self.max_steps = max_steps
        self.encode_hints = encode_hints
        self.device = device
        self.spec = spec
        self.no_feats = lambda x: x in no_feats or x.startswith("__")

        self.is_annealing_enabled = annealing
        self.annealing_state = 0

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
        if "c_h" in self.push_relabel_net.encoders:
            self.push_relabel_net.encoders.pop("c_h")

        self.global_relabel_net = Net(**net_args)  # ty:ignore[invalid-argument-type]
        for key in ["c_h"]:
            if key in self.global_relabel_net.encoders:
                self.global_relabel_net.encoders.pop(key)

        self.to(device)

    def op(self, trajectories, h_bfs, adj, phase):
        cand_bfs, h_bfs, h_preds_bfs = self.global_relabel_net.step(
            trajectories, h_bfs, adj
        )

        cand_pr, _, h_preds_pr = self.push_relabel_net.step(
            trajectories, h_bfs, adj
        )

        # ignore updates to flow from bfs net
        for key in ["f", "f_h", "e"]:
            if key in cand_bfs:
                cand_bfs[key] = torch.full_like(cand_bfs[key], clrs.OutputClass.MASKED)
            if key in h_preds_bfs:
                h_preds_bfs[key] = torch.full_like(
                    h_preds_bfs[key], clrs.OutputClass.MASKED
                )

        if self.decode_hints:
            hint_preds = {}
            for name, pred_pr in h_preds_pr.items():
                pred_bfs = h_preds_bfs.get(name, torch.zeros_like(pred_pr))

                bfs_mask_expanded = _match_phase(phase, pred_pr)
                pr_mask_expanded = 1.0 - bfs_mask_expanded

                hint_preds[name] = (pred_pr * pr_mask_expanded) + (
                    pred_bfs * bfs_mask_expanded
                )
        else:
            hint_preds = {}

        cand = {}
        for name, pred_bfs in cand_bfs.items():
            mask_cand = _match_phase(phase, pred_bfs)

            # Blend with PR outputs if available
            if name in cand_pr:
                pred_pr = cand_pr[name]
                cand[name] = (pred_pr * (1.0 - mask_cand)) + (pred_bfs * mask_cand)
            else:
                cand[name] = pred_bfs * mask_cand

        return cand, h_bfs, hint_preds

    def forward(self, features):
        output_preds = {}
        hint_preds = []

        num_steps = (
            self.max_steps if self.max_steps else features.hints[0].data.shape[0] - 1
        )
        batch_size, num_nodes = _dimensions(features.inputs)

        h_bfs = torch.zeros((batch_size, num_nodes, self.num_hidden)).to(self.device)
        adj = adj_mat(features).to(self.device)
        A = edge_attr_mat(features).to(self.device)

        def next_hint(i):
            use_teacher_forcing = self.training
            first_step = i == 0

            if self.is_annealing_enabled:
                self.annealing_state += 1
                use_teacher_forcing = use_teacher_forcing and not (
                    random() > (0.999**self.annealing_state)
                )

            if use_teacher_forcing or first_step:
                return _hints_i(features.hints, i)
            else:
                return _own_hints_i(last_valid_hints, self.spec, features, i)

        last_valid_hints = {
            hint.name: hint.data.to(self.device) for hint in next_hint(0)
        }

        for i in range(num_steps):
            trajectories = [features.inputs]
            if self.encode_hints:
                cur_hint = next_hint(i)
                trajectories.append(cur_hint)

            if self.decode_hints:
                phase = _get_phase(_hints_i(features.hints, i), self.device, batch_size)
            else:
                phase = torch.zeros((batch_size, 1)).to(self.device)

            cand, h_bfs, h_preds = self.op(trajectories, h_bfs, adj, phase)

            if "f" in cand:
                cand["f"] = cand["f"] * A

            if "f_h" in h_preds:
                h_preds["f_h"] = h_preds["f_h"] * A

            for name in last_valid_hints.keys():
                if self.no_feats(name):
                    continue

                if name in h_preds:
                    pred = h_preds[name]
                    _, _, type_ = self.spec[name]

                    if type_ == clrs.Type.MASK:
                        pred = torch.sigmoid(pred)
                    elif type_ in [
                        clrs.Type.MASK_ONE,
                        clrs.Type.CATEGORICAL,
                        clrs.Type.POINTER,
                    ]:
                        pred = torch.softmax(pred, dim=-1)
                    elif name in ["h", "e"]:
                        pred = torch.nn.functional.softplus(pred)

                    h_preds[name] = pred

                    if name == "h":
                        pred_for_hint = pred * num_nodes
                    else:
                        pred_for_hint = pred

                    is_masked = (h_preds[name] == clrs.OutputClass.MASKED) * 1.0
                    last_valid_hints[name] = (
                        is_masked * last_valid_hints[name]
                        + (1.0 - is_masked) * pred_for_hint
                    )

            hint_preds.append(h_preds)

            for name in cand:
                if name not in output_preds or features.lengths.sum() == 0:
                    output_preds[name] = cand[name]
                else:
                    is_not_done = is_not_done_broadcast(features.lengths, i, cand[name])

                    if name == "f":
                        phase_mask = _match_phase(phase, cand[name])
                        update_mask = is_not_done * (1.0 - phase_mask)
                    else:
                        update_mask = is_not_done

                    output_preds[name] = (
                        update_mask * cand[name]
                        + (1.0 - update_mask) * output_preds[name]
                    )

        return output_preds, hint_preds
