"""Experiment utilities for managing runs and experiments."""

import os
from dataclasses import dataclass
from datetime import datetime
from functools import partial
from pathlib import Path
from statistics import mean, stdev
from typing import Any, Callable, Dict, List

from clrs import Model
from ray.experimental import tqdm_ray
from torch.optim import Optimizer

from utils.io import dump


def set_seed(seed):
    """Set random seed for reproducibility."""
    import random

    import numpy as np
    import torch

    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def get_date():
    """Get current UTC date in ISO format."""
    return datetime.utcnow().isoformat()[:-7]


def random_search(space, num_samples):
    """Random hyperparameter search."""
    configs = []
    for _ in range(num_samples):
        c_ = {}
        for param_name, sample in space.items():
            c_[param_name] = sample()
        configs.append(c_)
    return configs


@dataclass(init=True, eq=True, frozen=True)
class Run:
    """A single experimental run with specific hyperparameters."""

    name: str
    seed: int
    batch_size: int
    model_fn: Callable[[Any], Model]
    config: Dict[str, Any]
    early_stop: bool
    early_stop_patience: int
    log_every: int
    train_steps: int
    verbose: bool


def init_runs(
    num_runs: int,
    hp_space: Dict,
    model_fn: Callable[[Any], Model],
    optim_fn: Callable[[Any], Optimizer],
    seed: int,
    batch_size: int,
    early_stop: bool,
    early_stop_patience: int,
    log_every: int,
    train_steps: int,
    verbose: bool,
):
    """Initialize experimental runs with random hyperparameter search."""

    assert "model" in hp_space, "Model's hyperparameters are missing"
    assert "optim" in hp_space, "Optimiser's hyperparameters are missing"

    model_configs = hp_space["model"]
    optim_configs = hp_space["optim"]

    search = partial(random_search, num_samples=num_runs)
    runs = list()

    seeds = [seed + i for i in range(num_runs)]

    for i, (m_conf, o_conf) in enumerate(
        zip(search(model_configs), search(optim_configs))
    ):
        model_fn_partial = partial(
            model_fn, **m_conf, optim_fn=partial(optim_fn, **o_conf)
        )

        run_ = Run(
            name=f"run_{i}",
            seed=seeds[i],
            config={**m_conf, **o_conf},
            model_fn=model_fn_partial,
            batch_size=batch_size,
            early_stop=early_stop,
            early_stop_patience=early_stop_patience,
            log_every=log_every,
            train_steps=train_steps,
            verbose=verbose,
        )

        runs.append(run_)

    return runs


def get_name():
    """Generate a random experiment name."""
    try:
        from randomname import generate

        return generate("v/programming", "n/algorithms")
    except ImportError:
        # Fallback if randomname is not installed
        import random
        import string

        return "".join(random.choices(string.ascii_lowercase, k=8))


def _init_ray(num_cpus: int, num_gpus: int, nw: int):
    """Initialize Ray for parallel execution."""
    import ray

    ray.init(num_cpus=num_cpus, num_gpus=num_gpus, include_dashboard=False)


NOT_DEFINED = "ND"


class Experiment:
    """Manages experimental runs with validation and testing."""

    def __init__(
        self,
        runs: List[Run],
        evaluate_fn: Callable,
        save_path: Path,
        name: str = "",
        num_cpus: int = 1,
        num_gpus: int = 0,
        num_valid_trials: int = 1,
        num_test_trials: int = 5,
        nw: int = 1,
        higher_is_better: bool = True,
    ):
        self.runs = runs
        self.evaluate_fn = evaluate_fn
        self.num_cpus = num_cpus
        self.num_gpus = num_gpus
        self.num_valid_trials = num_valid_trials
        self.num_test_trials = num_test_trials
        self.nw = nw
        self.higher_is_better = higher_is_better

        self.sequential = num_cpus == 1
        if not self.sequential:
            _init_ray(num_cpus, num_gpus, nw)

            from ray import remote

            @remote(num_cpus=1, num_gpus=1 / nw if num_gpus > 0 else 0)
            def run_valid_remote(*args, **kwargs) -> float:
                return run_valid(*args, **kwargs)

            self._fire = run_valid_remote
        else:
            self._fire = run_valid

        self.name = name + "-" + get_name()

        if (save_path / self.name).exists():
            print(f"[warning]: experiment {self.name} already existing.")
            self.name += "-new"


def _early_stop(loss, patience):
    """Check if early stopping criteria is met."""
    from math import isnan

    return isnan(loss) or patience <= 0


def run_valid(
    run: Run,
    evaluate_fn: Callable,
    tr_set,
    vl_set,
    save_path: Path,
    num_trials: int = 1,
    higher_is_better: bool = True,
) -> float:
    """Run validation for a single run configuration."""
    from rich.pretty import pprint as log

    set_seed(run.seed)

    def is_better(a, b):
        return a > b if higher_is_better else a < b

    log(run.config)

    HIGH_ = 1e4
    LOW_ = -HIGH_

    def closure():
        model = run.model_fn()
        best_score = LOW_ if higher_is_better else HIGH_
        patience_ = run.early_stop_patience
        losses, tr_scores, vl_scores = [], [], []

        for step in range(1, run.train_steps + 1):
            feedback = tr_set.next(run.batch_size)
            loss = model.feedback(feedback)
            losses.append(loss)

            if step % run.log_every == 0:
                tr_stats = evaluate_fn(
                    model,
                    feedback,
                    extras={"step": step, "loss": loss},
                    verbose=run.verbose,
                )
                vl_stats = evaluate_fn(
                    model, vl_set.next(), extras={"step": step}, verbose=run.verbose
                )

                tr_scores.append(tr_stats)
                vl_scores.append(vl_stats)

                log(
                    dict(
                        run_id=run.name,
                        **{f"tr_{key}": item for key, item in tr_stats.items()},
                        **{f"vl_{key}": item for key, item in vl_stats.items()},
                    )
                )

                if is_better(vl_stats["score"], best_score):
                    best_score = vl_stats["score"]
                else:
                    patience_ = (patience_ - 1) if run.early_stop else patience_

                if run.early_stop and _early_stop(tr_stats["loss"], patience_):
                    print("early stopping...")
                    break

        return losses, tr_scores, vl_scores, best_score

    losses, tr_scores, vl_scores, best_score = [], [], [], []

    for _ in range(num_trials):
        loss, tr_score, vl_score, score = closure()
        losses.append(loss)
        tr_scores.append(tr_score)
        vl_scores.append(vl_score)
        best_score.append(score)

    dump(
        dict(
            name=run.name,
            seed=run.seed,
            loss=losses,
            num_trials=num_trials,
            tr_scores=tr_scores,
            vl_scores=vl_scores,
            mean_score=mean(best_score),
            std_score=stdev(best_score) if num_trials > 1 else 0,
        ),
        save_path / run.name / "train_info.json",
    )

    dump(run.config, save_path / run.name / "parameters.json")

    return mean(best_score)


def _run_seq(exp, tr_set, vl_set, save_path):
    """Run experiments sequentially."""
    HIGH_ = 1e4
    LOW_ = -HIGH_
    best_score = LOW_ if exp.higher_is_better else HIGH_

    is_better = lambda a, b: a > b if exp.higher_is_better else a < b

    for run in exp.runs:
        score = run_valid(
            run,
            exp.evaluate_fn,
            tr_set,
            vl_set,
            save_path / exp.name,
            exp.num_valid_trials,
            exp.higher_is_better,
        )
        if is_better(score, best_score):
            best_score, best_run = score, run

    return best_score, best_run


def _run_par(exp, tr_set, vl_set, save_path):
    """Run experiments in parallel using Ray."""
    import ray

    remotes = {}

    for run in exp.runs:
        id_ = exp._fire.remote(
            run,
            tr_set,
            vl_set,
            save_path / exp.name,
            exp.num_valid_trials,
            exp.higher_is_better,
        )
        remotes[id_] = run

    HIGH_ = 1e4
    LOW_ = -HIGH_

    best_score = LOW_ if exp.higher_is_better else HIGH_
    is_better = lambda a, b: a > b if exp.higher_is_better else a < b

    for id_, run_ in remotes.items():
        score = ray.get(id_)
        if is_better(score, best_score):
            best_score, best_run = score, run_

        ray.cancel(id_, force=True)

    return best_score, best_run


def validate(experiment: Experiment, tr_set, vl_set, save_path: Path) -> Run:
    """Validate experiment runs and return best run."""

    if len(experiment.runs) == 1:
        best_run = experiment.runs[0]
        dump(
            {
                "meta": {
                    "name": best_run.name,
                    "vl_score": NOT_DEFINED,
                    "date": get_date(),
                },
                "config": best_run.config,
            },
            save_path / experiment.name / "best_run.json",
        )
        return best_run

    if experiment.sequential:
        best_score, best_run = _run_seq(experiment, tr_set, vl_set, save_path)
    else:
        best_score, best_run = _run_par(experiment, tr_set, vl_set, save_path)

    dump(
        {
            "meta": {"name": best_run.name, "vl_score": best_score, "date": get_date()},
            "config": best_run.config,
        },
        save_path / experiment.name / "best_run.json",
    )

    return best_run


def run_test(
    run: Run,
    evaluate_fn: Callable,
    tr_set,
    vl_set,
    ts_set,
    save_path: Path,
    num_trials: int = 5,
    higher_is_better: bool = True,
):
    """Run testing for a single run configuration."""
    from rich.pretty import pprint as log

    def is_better(a, b):
        return a > b if higher_is_better else a < b

    LOW_ = 0.0
    HIGH_ = 1e4

    if not save_path.exists():
        os.makedirs(save_path, exist_ok=True)

    def closure():
        model = run.model_fn()
        best = LOW_ if higher_is_better else HIGH_
        losses, tr_scores, vl_scores = [], [], []

        for step in tqdm_ray.tqdm(range(1, run.train_steps + 1)):
            feedback = tr_set.next(run.batch_size)
            loss = model.feedback(feedback)
            losses.append(loss)

            if step % run.log_every == 0:
                tr_stats = evaluate_fn(
                    model, feedback, extras={"step": step, "loss": loss}
                )
                vl_stats = evaluate_fn(
                    model, vl_set.next(), extras={"step": step}, verbose=run.verbose
                )

                tr_scores.append(tr_stats)
                vl_scores.append(vl_stats)

                log(
                    dict(
                        **{f"tr_{key}": item for key, item in tr_stats.items()},
                        **{f"vl_{key}": item for key, item in vl_stats.items()},
                    )
                )

                dump(tr_scores, save_path / "tr_scores.pkl")
                dump(vl_scores, save_path / "vl_scores.pkl")
                dump(loss, save_path / "loss.pkl")

                if is_better(vl_stats["score"], best):
                    best = vl_stats["score"]
                    print("do I ever reach here")
                    print(save_path)
                    dump(model.net_.state_dict(), save_path / f"model_{trial}.pth")

        return losses, tr_scores, vl_scores

    losses, tr_scores, vl_scores, ts_stats, ts_scores = [], [], [], [], []

    for trial in range(num_trials):
        loss, tr_score, vl_score = closure()
        losses.append(loss)
        tr_scores.append(tr_score)
        vl_scores.append(vl_score)

        model = run.model_fn()
        model.restore_model(save_path / f"model_{trial}.pth", "cpu")
        ts_stats.append(evaluate_fn(model, ts_set.next()))
        ts_scores.append(ts_stats[-1]["score"])

    return dict(
        loss=losses,
        num_trials=num_trials,
        tr_scores=tr_scores,
        vl_scores=vl_scores,
        ts_scores=ts_scores,
        ts_stats=ts_stats,
    )


def run_exp(experiment: Experiment, tr_set, vl_set, ts_set, save_path: Path):
    """Run full experiment: validation and testing."""
    save_path = save_path.resolve()

    best_run = validate(
        experiment=experiment, tr_set=tr_set, vl_set=vl_set, save_path=save_path
    )

    if experiment.sequential:
        res = run_test(
            run=best_run,
            evaluate_fn=experiment.evaluate_fn,
            tr_set=tr_set,
            vl_set=vl_set,
            ts_set=ts_set,
            num_trials=experiment.num_test_trials,
            higher_is_better=experiment.higher_is_better,
            save_path=save_path / experiment.name / "trials",
        )

        res["mean_score"] = mean(res["ts_scores"])
        res["std_score"] = (
            stdev(res["ts_scores"]) if experiment.num_test_trials > 1 else 0
        )
    else:
        import ray

        @ray.remote(num_cpus=1, num_gpus=1 / experiment.nw)
        def test(*args, **kwargs):
            return run_test(*args, **kwargs)

        remotes = []
        for i in range(experiment.num_test_trials):
            id_ = test.remote(
                run=best_run,
                evaluate_fn=experiment.evaluate_fn,
                tr_set=tr_set,
                vl_set=vl_set,
                ts_set=ts_set,
                num_trials=1,
                higher_is_better=experiment.higher_is_better,
                save_path=save_path / experiment.name / f"trial_{i}",
            )
            remotes.append(id_)

        res = {}
        for id_ in remotes:
            res_trial = ray.get(id_)
            if not res:
                res = res_trial
            else:
                for key in res.keys():
                    if not isinstance(res[key], list):
                        continue
                    res[key].extend(res_trial[key])

        res["mean_score"] = mean(res["ts_scores"])
        res["std_score"] = (
            stdev(res["ts_scores"]) if experiment.num_test_trials > 1 else 0
        )

    dump(res, save_path / experiment.name / "test_info.json")
