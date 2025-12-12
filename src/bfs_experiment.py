import clrs
import numpy as np
import torch
from functools import partial
from nn.models.epd import EncodeProcessDecode
from clrs._src import probing
from nn.models.impl import decoders
from clrs._src.samplers import _batch_io
from clrs._src.samplers import _batch_hints
from utils.data.graphs import two_community

_Stage = clrs.Stage
_Location = clrs.Location
_Type = clrs.Type

BFS_SINK_SPEC = {
    "pos": (_Stage.INPUT, _Location.NODE, _Type.SCALAR),
    "s": (_Stage.INPUT, _Location.NODE, _Type.MASK_ONE),
    "t": (_Stage.INPUT, _Location.NODE, _Type.MASK_ONE),
    "A": (_Stage.INPUT, _Location.EDGE, _Type.SCALAR),
    "adj": (_Stage.INPUT, _Location.EDGE, _Type.MASK),
    "h": (_Stage.HINT, _Location.NODE, _Type.SCALAR),
    "active_nodes": (_Stage.HINT, _Location.NODE, _Type.MASK),
    "h_out": (_Stage.OUTPUT, _Location.NODE, _Type.SCALAR),
}


def bfs_sink_algorithm(A: np.ndarray, s: int, t: int, transpose: bool = True):
    """
    Runs BFS from sink t to compute distance h.
    Returns trajectory.
    """
    n = A.shape[0]
    probes = probing.initialize(BFS_SINK_SPEC)

    h = np.full(n, n, dtype=int)
    h[t] = 0

    if transpose:
        adj_T = (A > 0).T.astype(float)
        A_T = A.T.copy()
    else:
        adj_T = (A > 0).astype(float)
        A_T = A.copy()

    probing.push(
        probes,
        clrs.Stage.INPUT,
        next_probe={
            "pos": np.arange(n) * 1.0 / n,
            "s": probing.mask_one(s, n),
            "t": probing.mask_one(t, n),
            "A": A_T,
            "adj": adj_T,
        },
    )

    q = [t]
    visited = np.zeros(n, dtype=bool)
    visited[t] = True

    # BFS Loop
    while len(q) > 0:
        active_mask = np.zeros(n, dtype=float)
        active_mask[q] = 1.0

        probing.push(
            probes,
            clrs.Stage.HINT,
            next_probe={
                "h": np.copy(h),
                "active_nodes": active_mask,
            },
        )

        new_q = []
        for v in q:
            for u in range(n):
                # Reverse edge: u -> v exists in residual graph if capacity(u,v) > 0
                if A[u, v] > 0 and not visited[u]:
                    visited[u] = True
                    h[u] = h[v] + 1
                    new_q.append(u)
        q = new_q

    active_mask = np.zeros(n, dtype=float)
    probing.push(
        probes,
        clrs.Stage.HINT,
        next_probe={
            "h": np.copy(h),
            "active_nodes": active_mask,
        },
    )

    probing.push(
        probes,
        clrs.Stage.OUTPUT,
        next_probe={
            "h_out": np.copy(h),
        },
    )

    probing.finalize(probes)
    return h, probes


class TrajSampler:
    """
    Sampler for generating synthetic graph trajectories for BFS experiments.

    Generates random graphs using the two-community model and computes
    BFS trajectories from a random source to a random sink.
    """

    def __init__(self, algorithm, spec, num_samples, num_nodes):
        self.algorithm = algorithm
        self.spec = spec
        self.num_samples = num_samples
        self.num_nodes = num_nodes
        self._ptr = 0
        self.rng = np.random.default_rng()

        self.graph_generator = two_community

    def next(self, batch_size=None):
        if batch_size is None:
            batch_size = 32

        inputs = []
        outputs = []
        hints = []
        lengths = []

        for _ in range(batch_size):
            if self.num_nodes == 16:
                n = self.rng.integers(8, 13) * 2
            else:
                n = self.num_nodes

            prob = max(0.35, 1.25 * np.log(n) / n)
            outer_prob = 0.05

            A = self.graph_generator(
                num_nodes=n,
                prob=prob,
                outer_prob=outer_prob,
                directed=True,
                weighted=True,
                rng=self.rng,
            )

            s = self.rng.choice(n // 2)
            t = self.rng.choice(range(n // 2 + 1, n))
            if s == t:
                t = (s + 1) % n

            h, probes = self.algorithm(A, s, t, transpose=True)

            inp, outp, hint = probing.split_stages(probes, self.spec)
            inputs.append(inp)
            outputs.append(outp)
            hints.append(hint)
            lengths.append(len(hint[0].data))

        batched_inputs = _batch_io(inputs)
        batched_outputs = _batch_io(outputs)
        batched_hints, batched_lengths = _batch_hints(hints, min_steps=0)

        for dp in batched_inputs:
            dp.data = torch.as_tensor(dp.data, dtype=torch.float32)
        for dp in batched_outputs:
            dp.data = torch.as_tensor(dp.data, dtype=torch.float32)
        for dp in batched_hints:
            dp.data = torch.as_tensor(dp.data, dtype=torch.float32)

        return clrs.Feedback(
            clrs.Features(batched_inputs, batched_hints, torch.tensor(batched_lengths)),
            batched_outputs,
        )


def main():
    print("Initializing BFS Experiment...")

    train_sampler = TrajSampler(
        algorithm=bfs_sink_algorithm, spec=BFS_SINK_SPEC, num_samples=1000, num_nodes=16
    )
    spec = BFS_SINK_SPEC

    model = EncodeProcessDecode(
        spec=spec,
        num_hidden=64,
        optim_fn=partial(torch.optim.Adam, lr=1e-3),
        dummy_trajectory=train_sampler.next(1),
        alpha=0.01,
        processor="pgn",
        aggregator="cat",
        decode_hints=True,
        encode_hints=True,
        max_steps=None,
    )

    print("Training...")
    for step in range(1000):
        feedback = train_sampler.next(1)
        loss = model.feedback(feedback)
        if step % 100 == 0:
            print(f"Step {step}: Loss = {loss:.6f}")

    print("Validation...")
    val_sampler = TrajSampler(
        algorithm=bfs_sink_algorithm,
        spec=BFS_SINK_SPEC,
        num_samples=100,
        num_nodes=32,
    )

    feedback = val_sampler.next(1)
    preds, (_, hint_preds) = model.predict(feedback.features)

    if len(feedback.outputs) > 0:
        gt_h = feedback.outputs[0].data
    else:
        print("GT h_out not found in output")
        return

    if "h_out" in preds:
        pred_h = preds["h_out"].data
    else:
        print("h_out not found in preds. Keys:", preds.keys())
        if len(hint_preds) > 0 and "h" in hint_preds[-1]:
            processed_hints = decoders.postprocess(hint_preds[-1], BFS_SINK_SPEC)
            pred_h = processed_hints["h"].data
        else:
            pred_h = torch.zeros_like(gt_h)

    # Inspect step-wise hints for sample 0
    print("Step-wise Hint Inspection (Sample 0)")

    gt_h_idx = -1
    for i, hint in enumerate(feedback.features.hints):
        if hint.name == "h":
            gt_h_idx = i
            break

    if gt_h_idx == -1:
        print("Error: GT h hint not found.")
    else:
        gt_h_traj = feedback.features.hints[gt_h_idx].data
        print(f"DEBUG: gt_h_traj shape: {gt_h_traj.shape}")
        current_n = gt_h_traj.shape[-1]

        num_steps = len(hint_preds)

        # fix shape
        is_time_major = False
        if gt_h_traj.shape[1] == 1 and gt_h_traj.shape[0] > 1:
            is_time_major = True

        steps_to_show = min(num_steps, 10)

        for t in range(steps_to_show):
            print(f"\nStep {t}:")

            # Target for step t is hint at t+1
            target_idx = t + 1

            valid_target = False
            if is_time_major:
                if target_idx < gt_h_traj.shape[0]:
                    gt_h_step = gt_h_traj[target_idx, 0].cpu().numpy()
                    valid_target = True
            else:
                if target_idx < gt_h_traj.shape[1]:
                    gt_h_step = gt_h_traj[0, target_idx].cpu().numpy()
                    valid_target = True

            if valid_target:
                print(f"  GT h:   {gt_h_step}")
            else:
                print("  GT h:   (End of trajectory)")

            # Pred at step t
            step_hints = hint_preds[t]
            if "h" in step_hints:
                processed = decoders.postprocess(
                    step_hints, BFS_SINK_SPEC, nb_nodes=current_n
                )
                pred_h_step = processed["h"].data[0].cpu().numpy()
                print(f"  Pred h: {pred_h_step}")
            else:
                print("  Pred h: (Missing)")

    print("\n--------------------------------------------")

    gt_h = gt_h.cpu().numpy()
    pred_h = pred_h.cpu().numpy()

    print("GT h (sample 0):", gt_h[0])
    print("Pred h (sample 0):", pred_h[0])

    mse = ((gt_h - pred_h) ** 2).mean().item()
    print(f"Validation MSE (avg over 100 samples): {mse:.4f}")


if __name__ == "__main__":
    main()
