from utils.data import load_dataset
from pathlib import Path


def main():
    print("Loading dataset...")
    try:
        train_sampler, spec = load_dataset(
            "train",
            "push_relabel_mincut",
            folder=Path("data/clrs/push_relabel_mincut/default"),
        )
    except Exception as e:
        print(f"Error loading dataset: {e}")
        print("Trying absolute path...")
        train_sampler, spec = load_dataset(
            "train",
            "push_relabel_mincut",
            folder=Path(
                "/Users/davidjgchen/Workspace/Stanford/CS224W/Project/data/clrs/push_relabel_mincut/default"
            ),
        )

    print("Fetching a trajectory...")
    feedback = train_sampler.next(1)
    features = feedback.features

    print("Available hints:", [h.name for h in features.hints])

    h_hint = None
    e_hint = None
    phase_hint = None

    for hint in features.hints:
        if hint.name == "h":
            h_hint = hint
        elif hint.name == "e":
            e_hint = hint
        elif hint.name == "__phase":
            phase_hint = hint

    if h_hint is None:
        print("Error: 'h' hint not found!")
        return

    num_steps = h_hint.data.shape[0]
    batch_size = h_hint.data.shape[1]
    num_nodes = h_hint.data.shape[2]

    print(
        f"Trajectory loaded. Steps: {num_steps}, Batch Size: {batch_size}, Nodes: {num_nodes}"
    )
    print("-" * 80)

    # We'll inspect the first sample in the batch
    b = 0

    for t in range(num_steps):
        h_val = h_hint.data[t][b]
        e_val = e_hint.data[t][b] if e_hint else None

        phase_val = -1
        if phase_hint:
            # Phase might be (T, B, 1) or (T, B)
            p_data = phase_hint.data[t][b]
            phase_val = p_data.item() if p_data.numel() == 1 else p_data[0].item()

        # Only print if h is non-zero or it's early in the trajectory
        if t < 20 or h_val.sum() > 0:
            phase_str = (
                "BFS"
                if phase_val == 1.0
                else ("PR" if phase_val == 0.0 else f"{phase_val}")
            )

            print(f"Step {t:03d} | Phase: {phase_str} (Raw: {phase_val})")
            print(f"  h: {h_val.tolist()}")
            if e_val is not None:
                print(f"  e: {e_val.tolist()}")
            print("-" * 40)


if __name__ == "__main__":
    main()
