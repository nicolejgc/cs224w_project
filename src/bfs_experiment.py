import clrs
import numpy as np
import torch
from functools import partial
from nn.models.epd import EncodeProcessDecode
from clrs._src import probing

# Define Spec
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
    "__phase": (_Stage.HINT, _Location.GRAPH, _Type.MASK), # Required for EPD
}

def bfs_sink_algorithm(A: np.ndarray, s: int, t: int):
    """
    Runs BFS from sink t to compute distance h.
    Returns trajectory.
    """
    n = A.shape[0]
    probes = probing.initialize(BFS_SINK_SPEC)

    h = np.full(n, n, dtype=int)
    h[t] = 0
    
    # Input probe
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

    q = [t]
    visited = np.zeros(n, dtype=bool)
    visited[t] = True

    # BFS Loop
    while len(q) > 0:
        # Push hint at start of step
        probing.push(
            probes,
            clrs.Stage.HINT,
            next_probe={
                "h": np.copy(h),
                "__phase": np.array([0.0]), # Dummy phase
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

    # Push final hint
    probing.push(
        probes,
        clrs.Stage.HINT,
        next_probe={
            "h": np.copy(h),
            "__phase": np.array([0.0]),
        },
    )

    # Output probe (h is also output, but we need to define it in spec if we want it as output)
    # Wait, spec defines h as HINT. Can it be OUTPUT too?
    # CLRS specs usually have separate names or same name if type matches.
    # Let's add 'h_out' as output or just rely on hint prediction?
    # The user said "output the 'h'".
    # Let's add 'h' as OUTPUT to spec as well?
    # CLRS allows same name for HINT and OUTPUT?
    # Let's check specs.py. "f" is in HINT and OUTPUT (as "f_h" and "f").
    # So let's keep "h" as HINT and add "h_out" as OUTPUT?
    # Or just use "h" for both?
    # If I use "h" for both, clrs might get confused or overwrite.
    # Let's use "h" for HINT and "h_out" for OUTPUT.
    
    # Actually, let's just use "h" as HINT and train on hints.
    # The user said "The task is to output the 'h'".
    # If we train on hints, we are training to predict h at every step.
    # This satisfies the requirement.
    
    probing.finalize(probes)
    return h, probes

class SimpleSampler:
    def __init__(self, algorithm, spec, num_samples, num_nodes):
        self.algorithm = algorithm
        self.spec = spec
        self.num_samples = num_samples
        self.num_nodes = num_nodes
        self._ptr = 0
        
    def next(self, batch_size=None):
        if batch_size is None:
            batch_size = 32 # Default batch size
            
        inputs = []
        outputs = []
        hints = []
        lengths = []
        
        for _ in range(batch_size):
            # Generate random graph
            # We need to generate A, s, t
            # For simplicity, let's use Erdős-Rényi G(n, p)
            n = self.num_nodes
            p = 0.5
            A = (np.random.rand(n, n) < p).astype(float)
            np.fill_diagonal(A, 0)
            
            s = np.random.randint(0, n)
            t = np.random.randint(0, n)
            while t == s:
                t = np.random.randint(0, n)
                
            # Run algorithm
            h, probes = self.algorithm(A, s, t)
            
            # Extract IO
            inp, outp, hint = probing.split_stages(probes, self.spec)
            inputs.append(inp)
            outputs.append(outp)
            hints.append(hint)
            lengths.append(len(hint[0].data)) # Length of trajectory
            
        # Batch IO
        from clrs._src.samplers import _batch_io
        from clrs._src.samplers import _batch_hints
        
        batched_inputs = _batch_io(inputs)
        batched_outputs = _batch_io(outputs)
        # _batch_hints requires min_steps argument (usually 0 or -1)
        # Let's try passing min_steps=0
        batched_hints, batched_lengths = _batch_hints(hints, min_steps=0)
        
        # Convert to torch
        for dp in batched_inputs:
             dp.data = torch.as_tensor(dp.data, dtype=torch.float32)
        for dp in batched_outputs:
             dp.data = torch.as_tensor(dp.data, dtype=torch.float32)
        for dp in batched_hints:
             dp.data = torch.as_tensor(dp.data, dtype=torch.float32)
             
        return clrs.Feedback(
            clrs.Features(batched_inputs, batched_hints, torch.tensor(batched_lengths)),
            batched_outputs
        )

def main():
    print("Initializing BFS Experiment...")
    
    # 1. Create Sampler
    train_sampler = SimpleSampler(
        algorithm=bfs_sink_algorithm,
        spec=BFS_SINK_SPEC,
        num_samples=1000,
        num_nodes=16
    )
    spec = BFS_SINK_SPEC
    
    # 2. Initialize Model
    model = EncodeProcessDecode(
        spec=spec,
        num_hidden=32,
        optim_fn=partial(torch.optim.Adam, lr=1e-3),
        dummy_trajectory=train_sampler.next(1),
        alpha=0.01,
        processor="pgn",
        aggregator="cat",
        decode_hints=True, # Enable hints to predict h at each step
        encode_hints=True, # Enable hints to use h as input
        max_steps=None, # Allow full unroll
    )
    
    print("Training...")
    for step in range(1000):
        feedback = train_sampler.next()
        loss = model.feedback(feedback)
        if step % 100 == 0:
            print(f"Step {step}: Loss = {loss:.4f}")
            
    print("Validation...")
    model.net_.eval()
    val_sampler = SimpleSampler(
        algorithm=bfs_sink_algorithm,
        spec=BFS_SINK_SPEC,
        num_samples=100,
        num_nodes=16
    )
    
    feedback = val_sampler.next(1)
    preds, (_, hint_preds) = model.predict(feedback.features)
    
    # Extract GT h from hints
    gt_h = None
    for hint in feedback.features.hints:
        if hint.name == "h":
            gt_h = hint.data
            break
            
    if gt_h is None:
        print("Error: GT h not found in hints.")
        return
    
    # Extract Pred h from hint_preds
    # hint_preds is a list of dicts (one per step)
    # We want the last step
    if len(hint_preds) > 0:
        last_step_hints = hint_preds[-1]
        if "h" in last_step_hints:
            pred_h = last_step_hints["h"]
            # pred_h might be raw logits or value.
            # For SCALAR, EPD decoder returns value.
            # But let's check if we need to postprocess.
            # decoders.postprocess handles masking etc.
            # But for simple scalar, it should be fine.
            # Actually, let's use decoders.postprocess on the hints to be safe
            from nn.models.impl import decoders
            processed_hints = decoders.postprocess(last_step_hints, BFS_SINK_SPEC)
            pred_h = processed_hints["h"].data
        else:
            print("Warning: 'h' not found in last_step_hints. Keys:", last_step_hints.keys())
            pred_h = torch.zeros_like(gt_h)
    else:
        print("Warning: hint_preds is empty.")
        pred_h = torch.zeros_like(gt_h)
    
    # Denormalize if needed (EPD decoders usually output normalized if scalar?)
    # Wait, clrs.process_preds handles this?
    # Let's check raw values first.
    
    # Move to CPU
    gt_h = gt_h.cpu().numpy()
    pred_h = pred_h.cpu().numpy()
    
    # Denormalize Pred h (EPD predicts normalized h/N)
    # Hint loss uses h/N.
    # So we should multiply by N.
    n = 16
    pred_h_denorm = pred_h * n
    
    print("GT h:", gt_h[0])
    print("Pred h (raw):", pred_h[0])
    print("Pred h (denorm):", pred_h_denorm[0])
    
    # Manual MSE
    mse = ((gt_h - pred_h_denorm)**2).mean().item()
    print(f"Validation MSE (denorm): {mse:.4f}")

if __name__ == "__main__":
    main()
