# GAUGE: Are Common Substructures Transferable? Riemannian Graph Foundation Model with Neural Vector Bundles [ICML 2026]

<p align="center">
  <a href="https://arxiv.org/abs/2606.03270">Paper</a> |
  <a href="#quick-start">Quick Start</a> |
  <a href="#citation">Citation</a>
</p>

GAUGE is a graph foundation model for learning transferable structural behavior across graphs. Instead of treating common substructures only as discrete motifs, GAUGE studies transferability through behavior invariance and connects it to intrinsic geometric flatness in the representation space.

The model builds a **Neural Vector Bundle** over a graph: each node is equipped with local coordinates, neighboring fibers are flattened when they are geometrically compatible, and a **Dirichlet loss** learns invariant structures while also measuring transfer effort during adaptation.

## Highlights

- A geometric view of structural transferability in graph pretraining, grounded in Riemannian geometry.
- A Neural Vector Bundle framework that models local graph geometry through learnable local coordinates and pseudo parallel transport.
- A pretrainable GAUGE architecture with Dirichlet loss for learning behavior-invariant substructures.
- Strong cross-domain few-shot transfer results on node-level and graph-level benchmarks, with additional validation on challenging settings such as graph isomorphism and zero-shot link prediction.

## Framework

<p align="center">
  <img src="./pics/framework.png" width="100%" alt="GAUGE framework">
</p>

<p align="center">
  <em>Figure 1. GAUGE constructs a neural vector bundle, learns invariant structures through Dirichlet energy, and transfers the pretrained model to downstream graphs.</em>
</p>

## Quick Start

### Environment

Install PyTorch and PyG according to your CUDA version first. The code also uses common graph-learning dependencies such as `torch_scatter`, `ogb`, `scikit-learn`, `networkx`, `numpy`, `tqdm`, and `pyyaml`.

### Pretraining

`--pretrain_graph_data` accepts one or more dataset names separated by spaces.

```shell
python main.py --run_type pretrain \
  --pretrain_graph_data ogbn-arxiv Reddit Questions Computers Amazon-ratings
```

This writes checkpoints to:

```text
checkpoints/pretrain/ogbn-arxiv_Reddit_Questions_Computers_Amazon-ratings/
```

You can also load an existing YAML configuration:

```shell
python main.py --run_type pretrain \
  --config_load_path scripts/pretrain/Computers_Amazon-ratings.yaml
```

### Few-Shot Adaptation

Use the same source dataset list as the pretrained checkpoint.

```shell
python main.py --run_type adapt \
  --pretrain_graph_data ogbn-arxiv Reddit Questions Computers Amazon-ratings \
  --pretrained_checkpoint checkpoints/pretrain/ogbn-arxiv_Reddit_Questions_Computers_Amazon-ratings/pretrain_epoch_20.pth \
  --data_name Photo \
  --task_type node_cls \
  --metric acc \
  --k_shot 5
```

Common options:

- `--task_type`: `node_cls`, `graph_cls`, or `link_cls`
- `--metric`: `acc`, `auc`, `ap`, `mse`, or `mae`
- `--k_shot`: number of labeled examples per class, for example `1` or `5`
- `--root`: dataset root directory, defaulting to `./datasets`

## Experimental Results

<p align="center">
  <img src="./pics/tab.png" width="100%" alt="Main cross-domain transfer results">
</p>

<p align="center">
  <em>Figure 2. Main cross-domain few-shot transfer results reported in the paper.</em>
</p>

## Visualization

<p align="center">
  <img src="./pics/tree.png" width="48%" alt="Invariant structures on tree graphs">
  <img src="./pics/grid.png" width="48%" alt="Invariant structures on grid graphs">
</p>

<p align="center">
  <img src="./pics/path.png" width="48%" alt="Invariant structures on path graphs">
  <img src="./pics/star.png" width="48%" alt="Invariant structures on star graphs">
</p>

<p align="center">
  <em>Figure 3. Invariant structures learned by GAUGE on tree, grid, path, and star graphs.</em>
</p>

## Repository Layout

```text
configs/      Command-line and YAML configuration parsers
cores/        GAUGE model, layers, losses, and pretraining trainer
data/         Dataset loading and preprocessing utilities
downstream/   Few-shot adaptation trainers and task heads
pics/         README figures and paper visualizations
scripts/      Example YAML configurations
```

## Citation
```bibtex
@inproceedings{
sun2026are,
title={Are Common Substructures Transferable? Riemannian Graph Foundation Model with Neural Vector Bundles},
author={Li Sun, Zhenhao Huang, Yiding Wang, Qin Chen, Pietro Li{\`o}, Philip S. Yu},
booktitle={Forty-third International Conference on Machine Learning},
year={2026}
}
```