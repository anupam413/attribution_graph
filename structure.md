# Attribution Graph Project — Folder Structure

```
attribution_graphs/
│
├── configs/
│   ├── model_config.yaml          # model name, device, dtype
│   ├── clt_config.yaml            # CLT hyperparams (n_features, lambda, c, lr)
│   └── graph_config.yaml          # pruning threshold, top-k logits, etc.
│
├── data/
│   ├── prompts/
│   │   ├── factual_recall.txt     # one prompt per line for experiments
│   │   ├── addition.txt
│   │   └── custom.txt
│   └── activation_cache/          # cached MLP in/out tensors for CLT training
│       └── .gitkeep
│
├── src/
│   ├── __init__.py
│   │
│   ├── model/
│   │   ├── __init__.py
│   │   ├── loader.py              # ← PHASE 2: HookedTransformer setup (THIS FILE)
│   │   └── hooks.py               # reusable hook utilities
│   │
│   ├── transcoder/
│   │   ├── __init__.py
│   │   ├── per_layer.py           # Phase 3a: per-layer transcoder
│   │   ├── cross_layer.py         # Phase 3b: CLT
│   │   └── train.py               # training loop, loss functions
│   │
│   ├── replacement_model/
│   │   ├── __init__.py
│   │   └── local.py               # Phase 3c: local replacement model
│   │
│   ├── attribution/
│   │   ├── __init__.py
│   │   ├── jacobian.py            # Phase 3d: backward Jacobians, edge weights
│   │   ├── graph.py               # graph construction (nodes + edges)
│   │   └── prune.py               # Phase 3e: influence matrix + pruning
│   │
│   └── utils/
│       ├── __init__.py
│       ├── cache.py               # activation caching helpers
│       └── viz.py                 # simple graph printing / inspection
│
├── experiments/
│   ├── factual_recall.py          # reproduce Michael Jordan case study
│   ├── addition.py                # reproduce 36+59 case study
│   └── custom_prompt.py           # template for your own experiments
│
├── notebooks/
│   ├── 01_model_exploration.ipynb
│   ├── 02_transcoder_training.ipynb
│   ├── 03_attribution_graphs.ipynb
│   └── 04_validation.ipynb
│
├── tests/
│   ├── test_loader.py
│   ├── test_transcoder.py
│   └── test_attribution.py
│
├── requirements.txt
├── setup.py
└── README.md
```

## File responsibilities at a glance

| File | Phase | What it does |
|------|-------|--------------|
| `src/model/loader.py` | 2 | Loads model, exposes hooks, caches activations |
| `src/transcoder/per_layer.py` | 3a | JumpReLU transcoder, one per MLP layer |
| `src/transcoder/cross_layer.py` | 3b | CLT: layer-ℓ features decode to all ℓ'≥ℓ |
| `src/transcoder/train.py` | 3a/b | MSE + sparsity loss, training loop |
| `src/replacement_model/local.py` | 3c | Freezes attn patterns, substitutes MLPs, adds error nodes |
| `src/attribution/jacobian.py` | 3d | Stop-gradient Jacobians, virtual weights |
| `src/attribution/graph.py` | 3d | Assembles node/edge graph from Jacobian output |
| `src/attribution/prune.py` | 3e | Indirect influence matrix, threshold-based pruning |
