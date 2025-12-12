# Instructions for running the code!

## Setting up the environment

We used uv, so a simple `uv synv` should work. The code should work well on Apple Silicon, but it has not been tested in other environments.

## Generating the data

You will need to generate the datasets. This can be done with the command:
```
uv run src/build_data.py main push_relabel_mincut
```

The VesselGraph datasets can be downloaded from https://github.com/jocpae/VesselGraph.

## Training and testing on synthetic data

In order to train the model, you will need to create a configuration file and run a command similar to the following one:

```
uv run --active src/run.py valid src/config/exp/prmc.davidtestquick.yaml data/clrs/push_relabel_mincut/default --model pr_net --aggregator cat
```

To test the model after training has completed, find the relevant file (in `src/runs/`) that contains the saved model weights, and run the following commands based on the specific test set name you wish to test against:

```
uv run --active src/run.py test push_relabel_mincut src/runs/[model folder] data/clrs/push_relabel_mincut/default --test-set test_2x
```

## VesselGraph transfer learning

TODO

# Walkthrough of the codebase:

Much of the code-base is inspired by the [Dual Algorithmic Reasoning codebase](github.com/danilonumeroso/dar), and some implementations (such as the baseline models) are sourced from their codebase with minimal modification.

Configuration files for running experiments can be found in `src/config`, source code for models can be found in `src/nn`, and other code relating to the experiment can be found in `src/util`.

For example, code for graph generation can be found in `src/build_data.py` and `src/utils/data/graphs.py`.
Code for algorithm implementations can be found in `src/utils/data/algorithms`.