"""
Push-Relabel Neural Network (PR_Net)

A specialized neural network for learning the Push-Relabel max-flow algorithm.
Unlike Ford-Fulkerson (which alternates BFS + flow update), Push-Relabel uses:
- Local push/relabel operations on nodes with excess flow
- Periodic global relabeling via BFS to update node heights

This network has two sub-networks:
- push_relabel_net: Learns push and relabel operations
- global_relabel_net: Learns the BFS-based height update
"""

from random import random
from typing import Callable, Dict, List

import clrs
import torch

from nn import losses as loss
from nn.models.epd import EncodeProcessDecode_Impl as Net
from nn.models.impl import (
    _dimensions,
    _expand_to,
    _get_fts,
    _hints_i,
    decoders,
)
from utils import is_not_done_broadcast
from utils.data import adj_mat, edge_attr_mat

Result = Dict[str, clrs.DataPoint]

_Feedback = clrs.Feedback
_Spec = clrs.Spec
_Stage = clrs.Stage
_Location = clrs.Location
_Type = clrs.Type
_DataPoint = clrs.DataPoint


def _phase_mask(hints):
    """Extract the phase hint (0 = push-relabel, 1 = BFS)."""
    for hint in hints:
        if hint.name == "phase":
            return hint
    return None


class PR_Net(clrs.Model):
    """
    Push-Relabel Neural Network
    
    Learns to predict:
    - Max flow (f) on edges
    - Min cut (c) partition of nodes
    
    By learning from:
    - Height labels (h) - which nodes are "higher"
    - Excess flow (e) - which nodes have excess to push
    - Active nodes - which nodes are currently being processed
    - Phase - whether we're in push-relabel or global-relabel (BFS) phase
    """
    
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

        self.device = "cpu"
        if torch.cuda.is_available():
            self.device = "cuda"
        elif torch.mps.is_available():
            self.device = "mps"

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
        self.encode_hints = encode_hints
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

        total_loss: float | torch.Tensor = 0.0
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

        if not isinstance(total_loss, torch.Tensor):
            return 0.0

        total_loss.backward()
        self.optimizer.step()

        return total_loss.item()

    def feedback(self, feedback: _Feedback) -> float:
        loss = self._train_step(feedback)
        return loss

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
            losses["aux_" + truth.name] = (
                loss.hint_loss(aux_preds, truth, feedback, self.alpha, self.device)
                .cpu()
                .item()
            )
            total_loss += losses["aux_" + truth.name]

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
    """
    Implementation of the Push-Relabel Network.
    
    Architecture:
    - push_relabel_net: Handles local push/relabel operations
      - Learns to predict: h (height), e (excess), f_h (flow), active_nodes
      - Used when phase = PUSH_RELABEL (0)
    
    - global_relabel_net: Handles BFS-based global relabeling
      - Learns to predict: h (height updates from BFS)
      - Used when phase = BFS (1)
    
    The phase hint tells us which operation the algorithm is performing,
    and we route to the appropriate sub-network.
    """
    
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
        self.no_feats = lambda x: x in no_feats or x.startswith("__")

        # Sub-network for push/relabel operations
        # This network learns the local node operations:
        # - When to push flow (based on height and excess)
        # - When to relabel (increase height)
        self.push_relabel_net = Net(
            spec,
            dummy_trajectory,
            num_hidden,
            encode_hints,
            decode_hints,
            processor,
            aggregator,
            no_feats,
            add_noise=add_noise,
            device=device,
        )

        # Sub-network for global relabeling (BFS phase)
        # This network learns the BFS traversal for height updates
        self.global_relabel_net = Net(
            spec,
            dummy_trajectory,
            num_hidden,
            encode_hints,
            decode_hints,
            processor,
            aggregator,
            no_feats,
            add_noise=add_noise,
            device=device,
        )

        # Remove encoders that aren't relevant for each sub-network
        # Push-relabel doesn't need to encode cut hints (that's output)
        if "c_h" in self.push_relabel_net.encoders:
            del self.push_relabel_net.encoders["c_h"]
        
        # Global relabel (BFS) focuses on height updates
        # It doesn't need to predict flow or cut directly
        for key in ["f_h", "c_h"]:
            if key in self.global_relabel_net.encoders:
                del self.global_relabel_net.encoders[key]

        self.device = device
        self.spec = spec
        self.encode_hints = encode_hints

        self.to(device)

    def op(self, trajectories, h_pr, adj, phase):
        """
        Perform one step of the network based on current phase.
        
        Args:
            trajectories: Input features and hints
            h_pr: Hidden state from push-relabel net
            adj: Adjacency matrix
            phase: Current phase (0 = push-relabel, 1 = BFS)
        
        Returns:
            cand: Candidate output predictions
            h_pr: Updated hidden state
            hint_preds: Predicted hints
        """
        # Get predictions from push-relabel network
        _, h_pr, h_preds_pr = self.push_relabel_net.step(trajectories, h_pr, adj)
        
        # Get predictions from global-relabel (BFS) network
        cand, _, h_preds_bfs = self.global_relabel_net.step(
            trajectories, h_pr.detach(), adj
        )

        # Mask out irrelevant predictions based on phase
        # During push-relabel phase, BFS predictions are masked
        # During BFS phase, push-relabel predictions are masked
        
        if self.decode_hints:
            hint_preds = {}
            
            # phase = 0 means push-relabel, phase = 1 means BFS
            # nonzero() returns [N, 1], squeeze to [N] for proper indexing
            idx_pr = (1 - phase).flatten().nonzero().squeeze(-1)  # Push-relabel indices
            idx_bfs = phase.flatten().nonzero().squeeze(-1)        # BFS indices
            
            for name in h_preds_pr.keys():
                hint_preds[name] = torch.empty_like(h_preds_pr[name]).to(self.device)
                hint_preds[name].fill_(clrs.OutputClass.MASKED)
                
                # Use push-relabel predictions when in push-relabel phase
                if idx_pr.numel() > 0:
                    hint_preds[name][idx_pr] = h_preds_pr[name][idx_pr]
                # Use BFS predictions when in BFS phase
                if idx_bfs.numel() > 0:
                    hint_preds[name][idx_bfs] = h_preds_bfs[name][idx_bfs]

        # Reset hidden state when transitioning phases
        mask_1d = phase.flatten().bool()
        mask_expanded = _expand_to(phase.bool(), len(h_pr.shape))
        h_pr = h_pr.masked_fill(mask_expanded, 0)

        # Mask output candidates based on phase
        for name in cand.keys():
            n_dims = len(cand[name].shape)
            mask = _expand_to(phase.bool(), n_dims)
            cand[name] = cand[name].masked_fill(mask, clrs.OutputClass.MASKED)

        return cand, h_pr, hint_preds

    def forward(self, features):
        """
        Forward pass through the Push-Relabel network.
        
        Processes the algorithm execution trace step by step,
        using phase information to route to appropriate sub-network.
        """
        output_preds = {}
        hint_preds = []

        num_steps = (
            self.max_steps if self.max_steps else features.hints[0].data.shape[0] - 1
        )
        batch_size, num_nodes = _dimensions(features.inputs)

        # Initialize hidden state
        h_pr = torch.zeros((batch_size, num_nodes, self.num_hidden)).to(self.device)

        adj = adj_mat(features).to(self.device)
        A = edge_attr_mat(features).to(self.device)

        for i in range(num_steps):
            # Get current hints
            trajectories = [features.inputs]
            if self.encode_hints:
                cur_hint = _hints_i(features.hints, i)
                trajectories.append(cur_hint)

            # Get phase for this step
            if self.decode_hints:
                phase_hint = _phase_mask(_hints_i(features.hints, i))
                if phase_hint is not None:
                    phase_data = phase_hint.data.to(self.device)
                    # phase_data is one-hot: [batch, 2] where [1,0]=push-relabel, [0,1]=BFS
                    # Convert to scalar index: 0=push-relabel, 1=BFS
                    phase = phase_data.argmax(dim=-1, keepdim=True).float()  # [batch, 1]
                else:
                    # Default to push-relabel phase if no phase hint
                    phase = torch.zeros((batch_size, 1)).to(self.device)
            else:
                phase = torch.zeros((batch_size, 1)).to(self.device)

            # Run the appropriate sub-network
            cand, h_pr, h_preds = self.op(trajectories, h_pr, adj, phase)

            # Apply capacity constraint to flow predictions
            if "f" in cand:
                idx_pr = (1 - phase).flatten().nonzero().squeeze(-1)
                if idx_pr.numel() > 0:
                    cand["f"][idx_pr] = (A * cand["f"])[idx_pr]

            if "f_h" in h_preds:
                idx_pr = (1 - phase).flatten().nonzero().squeeze(-1)
                if idx_pr.numel() > 0:
                    h_preds["f_h"][idx_pr] = (A * h_preds["f_h"])[idx_pr]

            hint_preds.append(h_preds)

            # Update output predictions
            for name in cand:
                if name not in output_preds or features.lengths.sum() == 0:
                    output_preds[name] = cand[name]
                else:
                    is_not_done = is_not_done_broadcast(features.lengths, i, cand[name])
                    is_pr_phase = (1 - phase)  # Push-relabel phase
                    mask = is_not_done * _expand_to(is_pr_phase, len(is_not_done.shape))
                    output_preds[name] = (
                        mask * cand[name] + (1.0 - mask) * output_preds[name]
                    )

        return output_preds, hint_preds

