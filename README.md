# cs224w_project

Currently this thing kind of works:

```
uv run --active src/run.py valid src/config/exp/pr.davidtest2.yaml data/clrs/push_relabel/default --model epd --aggregator sum --num-test-trials 2
```

```
uv run --active src/run.py valid src/config/exp/ffmc.davidtest.yaml data/clrs/ford_fulkerson_mincut/default --model mf_net
```