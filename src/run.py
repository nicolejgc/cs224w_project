import os
from functools import partial
from pathlib import Path
from statistics import mean, stdev

import clrs
import torch
import typer

from config.hyperparameters import HP_SPACE
from nn.models import EncodeProcessDecode, MF_Net, MF_NetPipeline, PR_Net
from utils.data import load_dataset
from utils.experiment_utils import Experiment, init_runs, run_exp, set_seed
from utils.experiments import evaluate
from utils.io import dump, load
from utils.types import Algorithm

app = typer.Typer(add_completion=False, pretty_exceptions_show_locals=False)


def choose_default_dtype(name: str):
    assert name in ["float16", "float32"]

    if name == "float16":
        return torch.float16

    return torch.float32


def choose_model(name: str):
    assert name in ["epd", "mf_net", "mf_net_pipe", "mf_net_res", "pr_net"]

    if name == "epd":
        model_class = EncodeProcessDecode
    elif name == "mf_net":
        model_class = MF_Net
    elif name == "pr_net":
        model_class = PR_Net
    else:
        model_class = MF_NetPipeline

    return model_class


def choose_hint_mode(mode: str):
    assert mode in ["io", "o", "none"]

    if mode == "io":
        encode_hints, decode_hints = True, True
    elif mode == "o":
        encode_hints, decode_hints = False, True
    elif mode == "none":
        encode_hints, decode_hints = False, False

    return encode_hints, decode_hints


def split_probes(feedback):
    _, source, tail, weights, adj = feedback.features.inputs
    return (
        source.data.squeeze().numpy().argmax(-1),
        tail.data.squeeze().numpy().argmax(-1),
        weights.data.squeeze().numpy(),
        adj.data.squeeze().numpy(),
    )


def _preprocess_yaml(config):
    assert "algorithm" in config.keys()
    assert "runs" in config.keys()
    assert "experiment" in config.keys()

    for key in config["runs"].keys():
        if key == "hp_space":
            config["runs"][key] = HP_SPACE[config["runs"][key]]
            continue

    return config


@app.command()
def valid(
    exp_path: Path,
    data_path: Path,
    model: str = "epd",
    hint_mode: str = "io",
    max_steps: int | None = None,
    num_cpus: int | None = None,
    num_gpus: int = 1,
    nw: int = 4,
    no_feats: str = "adj",
    noise: bool = False,
    processor: str = "pgn",
    aggregator: str = "max",
    save_path: Path = Path("src/runs"),
    dtype: str = "float32",
    num_test_trials: int = 4,
    seed: int | None = None,
):
    torch.set_default_dtype(choose_default_dtype(dtype))

    assert aggregator in ["max", "sum", "mean", "cat"]
    assert processor in ["mpnn", "pgn"]

    if seed is None:
        seed = int.from_bytes(os.urandom(2), byteorder="big")

    encode_hints, decode_hints = choose_hint_mode(hint_mode)
    model_class = choose_model(model)

    configs = _preprocess_yaml(load(exp_path))

    alg = configs["algorithm"]
    set_seed(seed)

    print("loading val...")
    vl_sampler, _ = load_dataset("val", alg, folder=data_path)
    print("loading test...")
    ts_sampler, _ = load_dataset("test", alg, folder=data_path)
    print("loading tr...")
    tr_sampler, spec = load_dataset("train", alg, folder=data_path)

    print("loading done")

    model_fn = partial(
        model_class,
        spec=spec,
        dummy_trajectory=tr_sampler.next(1),
        decode_hints=decode_hints,
        encode_hints=encode_hints,
        add_noise=noise,
        no_feats=no_feats.split(","),
        max_steps=max_steps,
        processor=processor,
        aggregator=aggregator,
    )

    runs = init_runs(
        seed=seed, model_fn=model_fn, optim_fn=torch.optim.SGD, **configs["runs"]
    )

    experiment = Experiment(
        runs=runs,
        evaluate_fn=evaluate,
        save_path=save_path,
        num_cpus=num_cpus if num_cpus else num_gpus * nw,
        num_gpus=num_gpus,
        nw=nw,
        num_test_trials=num_test_trials,
        **configs["experiment"],
    )

    dump(
        dict(
            alg=alg,
            data_path=str(data_path),
            hint_mode=hint_mode,
            model=model,
            aggregator=aggregator,
            processor=processor,
            no_feats=no_feats.split(","),
            seed=seed,
        ),
        save_path / experiment.name / "config.json",
    )

    print(f"Experiment name: {experiment.name}")

    run_exp(
        experiment=experiment,
        tr_set=tr_sampler,
        vl_set=vl_sampler,
        ts_set=ts_sampler,
        save_path=save_path,
    )


@app.command()
def test(
    alg: Algorithm,
    test_path: Path,
    data_path: Path,
    max_steps: int | None = None,
    test_set: str = "test",
):
    from utils.metrics import eval_categorical, masked_mae

    ts_sampler, spec = load_dataset(test_set, alg.value, folder=data_path)
    best_run = load(test_path / "best_run.json")["config"]
    config = load(test_path / "config.json")

    hint_mode = config["hint_mode"]

    encode_hints, decode_hints = choose_hint_mode(hint_mode)
    model_class = choose_model(config["model"])

    feedback = ts_sampler.next()
    runs = []

    adj = feedback.features.inputs[-2].data.numpy()

    def predict(features, outputs, i):
        model = model_class(
            spec=spec,
            dummy_trajectory=ts_sampler.next(1),
            num_hidden=best_run["num_hidden"],
            alpha=best_run["alpha"],
            aggregator=config["aggregator"],
            processor=config["processor"],
            max_steps=max_steps,
            no_feats=config["no_feats"],
            decode_hints=decode_hints,
            encode_hints=encode_hints,
            optim_fn=torch.optim.Adam,
        )
        model.restore_model(test_path / f"trial_{i}" / "model_0.pth", "cuda")

        preds, aux = model.predict(features)
        for key in preds:
            preds[key].data = preds[key].data.cpu()

        metrics = {}
        for truth in feedback.outputs:
            type_ = preds[truth.name].type_
            y_pred = preds[truth.name].data.numpy()
            y_true = truth.data.numpy()

            if type_ == clrs.Type.SCALAR:
                metrics[truth.name] = masked_mae(y_pred, y_true * adj).item()

            elif type_ == clrs.Type.CATEGORICAL:
                metrics[truth.name] = eval_categorical(y_pred, y_true).item()

        dump(preds, test_path / f"trial_{i}" / f"preds_{i}.{test_set}.pkl")
        dump(
            model.net_.flow_net.h_t.cpu(),
            test_path / f"trial_{i}" / f"H{i}.{test_set}.pth",
        )
        dump(
            model.net_.flow_net.edge_attr.cpu(),
            test_path / f"trial_{i}" / f"E{i}.{test_set}.pth",
        )
        return metrics

    for i in range(5):
        if not (test_path / f"trial_{i}" / "model_0.pth").exists():
            continue

        runs.append(predict(feedback.features, feedback.outputs, i))
        torch.cuda.empty_cache()
        torch.mps.empty_cache()

    dump(runs, test_path / f"scores.{test_set}.json")

    for key in runs[0]:
        out = [evals[key] for evals in runs]
        print(key, mean(out), "pm", stdev(out) if len(out) > 1 else 0)


@app.command()
def transfer(
    checkpoint_path: str = typer.Argument(..., help="Path to trained model checkpoint (.pth)"),
    config_path: str = typer.Argument(..., help="Path to experiment config (e.g., ffmc_vessel.yaml)"),
    data_path: str = typer.Argument(..., help="Path to vessel feature data"),
    phase: str = typer.Option("2", help="Transfer phase: 2=replace encoders, 3=freeze+train, 4=finetune"),
    model: str = typer.Option("mf_net", help="Model type"),
    vessel_features: str = typer.Option("length,distance,curvedness", help="Comma-separated vessel features"),
    save_path: str = typer.Option("src/runs/transfer", help="Output directory"),
    num_cpus: int = typer.Option(1, help="Number of CPUs"),
    freeze_backbone: bool = typer.Option(True, help="Freeze backbone (Phase 3)"),
    lr: float = typer.Option(1e-3, help="Learning rate"),
):
    """
    Transfer learning pipeline for DAR.
    
    This implements the transfer learning from the DAR paper:
    - Phase 2: Replace capacity encoder with vessel feature encoders
    - Phase 3: Freeze backbone, train new encoders on synthetic data
    - Phase 4: Fine-tune on real brain vessel data
    
    Example:
        # Phase 2+3: Replace encoders and train with frozen backbone
        uv run src/run.py transfer ./src/runs/ffmc/best_model.pth \\
            src/config/exp/ffmc_vessel.yaml \\
            src/data/vessel/default \\
            --phase 3 --freeze-backbone
        
        # Phase 4: Fine-tune everything
        uv run src/run.py transfer ./src/runs/transfer/model.pth \\
            src/config/exp/ffmc_vessel.yaml \\
            src/data/vessel_real/default \\
            --phase 4 --no-freeze-backbone --lr 1e-4
    """
    from utils.transfer import replace_encoder, freeze_model, unfreeze_model
    
    save_dir = Path(save_path)
    save_dir.mkdir(parents=True, exist_ok=True)
    
    vessel_feature_list = vessel_features.split(",")
    config = load(config_path)
    algorithm = Algorithm(config["algorithm"])
    
    print("\n" + "=" * 70)
    print("DAR TRANSFER LEARNING")
    print("=" * 70)
    print(f"Checkpoint: {checkpoint_path}")
    print(f"Phase: {phase}")
    print(f"Vessel features: {vessel_feature_list}")
    print(f"Freeze backbone: {freeze_backbone}")
    
    # Load data with vessel features
    print("\nLoading data...")
    _, val, test, tr = load_dataset(data_path, algorithm=algorithm)
    dummy_trajectory = val[0]
    
    # Get spec for vessel algorithm
    from utils.data.algorithms.specs import SPECS
    spec = SPECS[algorithm.value]
    
    # Create model
    model_class = choose_model(model)
    hp = HP_SPACE[config["runs"]["hp_space"]]()
    
    # Load trained model weights
    print(f"\nLoading checkpoint: {checkpoint_path}")
    
    # Recreate model architecture
    net = model_class(
        spec=spec,
        num_hidden=hp["num_hidden"],
        optim_fn=partial(torch.optim.Adam, lr=lr),
        dummy_trajectory=dummy_trajectory,
        alpha=hp.get("alpha", 0),
        processor=hp["processor"],
        aggregator=hp["aggregator"],
        decode_hints=True,
        encode_hints=True,
    )
    
    # Load weights (with strict=False to allow missing/extra keys)
    checkpoint = torch.load(checkpoint_path, map_location='cpu')
    try:
        net.net_.load_state_dict(checkpoint, strict=False)
        print("Loaded checkpoint successfully")
    except Exception as e:
        print(f"Warning loading checkpoint: {e}")
        print("Attempting to load with relaxed matching...")
    
    # =========================================================================
    # PHASE 2: Replace encoders
    # =========================================================================
    if phase in ["2", "3", "all"]:
        print("\n" + "-" * 70)
        print("PHASE 2: Replacing capacity encoder with vessel features")
        print("-" * 70)
        
        # Check current encoders
        print("\nEncoders BEFORE replacement:")
        for name in net.net_.net.encoders.keys():
            print(f"  - {name}")
        
        # Replace A (capacity) with vessel features
        replace_encoder(
            net,
            old_encoder_name="A",
            new_encoder_names=vessel_feature_list
        )
        
        print("\nEncoders AFTER replacement:")
        for name in net.net_.net.encoders.keys():
            print(f"  - {name}")
    
    # =========================================================================
    # PHASE 3: Freeze backbone, train new encoders
    # =========================================================================
    if phase in ["3", "all"] and freeze_backbone:
        print("\n" + "-" * 70)
        print("PHASE 3: Freezing backbone for encoder training")
        print("-" * 70)
        
        exclude_patterns = [f"encoders.{name}" for name in vessel_feature_list]
        freeze_model(net, exclude=exclude_patterns)
        
        # Recreate optimizer with only trainable params
        net.optimizer = torch.optim.Adam(
            filter(lambda p: p.requires_grad, net.parameters()),
            lr=lr
        )
    
    # =========================================================================
    # PHASE 4: Fine-tune (unfreeze if needed)
    # =========================================================================
    if phase == "4":
        print("\n" + "-" * 70)
        print("PHASE 4: Fine-tuning")
        print("-" * 70)
        
        if not freeze_backbone:
            unfreeze_model(net)
            print("All parameters unfrozen for fine-tuning")
        
        net.optimizer = torch.optim.Adam(
            filter(lambda p: p.requires_grad, net.parameters()),
            lr=lr
        )
    
    # =========================================================================
    # Training
    # =========================================================================
    print("\n" + "-" * 70)
    print("Starting training...")
    print("-" * 70)
    
    # Create experiment
    runs = init_runs(
        config["runs"],
        HP_SPACE[config["runs"]["hp_space"]],
        val,
        spec,
        model_class,
    )
    
    # Override the model in runs with our modified model
    for run in runs:
        run.model = net
    
    experiment = Experiment(
        name=config["experiment"]["name"] + "-transfer",
        runs=runs,
        score_fn=lambda m: m["vl_score"],
        maximize_score=config["experiment"]["higher_is_better"],
        model_class=model_class,
    )
    
    run_exp(
        experiment,
        n_trials=config["experiment"]["num_valid_trials"],
        num_workers=num_cpus,
        tr=tr,
        test=test,
        save_path=save_dir,
    )
    
    # Save final model
    final_path = save_dir / "transfer_model.pth"
    torch.save(net.net_.state_dict(), final_path)
    print(f"\nSaved transfer model to: {final_path}")


if __name__ == "__main__":
    app()
