[pypi-image]: https://badge.fury.io/py/torch-geometric.svg
[pypi-url]: https://pypi.org/project/torch-quiver/

<p align="center">
  <img height="150" src="docs/multi_medias/imgs/quiver-logo-min.png" />
</p>

--------------------------------------------------------------------------------

Quiver is a distributed graph learning library for [PyTorch Geometric](https://github.com/pyg-team/pytorch_geometric) (PyG). The goal of Quiver is to make distributed graph learning easy-to-use and achieve high-performance.

The following documents additional code added to complete the **Scalability Opportunities in Graph Learning
for Singly Large Graph Datasets** project. All rights reserved to the Quiver team, and the original repo [Quiver](https://torch-quiver.readthedocs.io/en/latest/?badge=latest)

[![Documentation Status](https://readthedocs.org/projects/torch-quiver/badge/?version=latest)](https://torch-quiver.readthedocs.io/en/latest/?badge=latest)


<!-- **Quiver** is a high-performance GNN training add-on which can fully utilize the hardware to achive the best GNN trainning performance. By integrating Quiver into your GNN training pipeline with **just serveral lines of code change**, you can enjoy **much better end-to-end performance** and **much better scalability with multi-gpus**, you can even achieve **super linear scalability** if your GPUs are connected with NVLink, Quiver will help you make full use of NVLink. -->

--------------------------------------------------------------------------------

## Adapted:

The following code was used in each section:

### Scalability: Larger Models

```cmd
$ examples/multi_gpu/pyg/reddit/dist_sampling_ogb_reddit_pyg_gpipe.py
```
This script contains the updated code on experiments using GPipe and traditional DL model parallleism.

```cmd
$ examples/multi_gpu/pyg/reddit/dist_sampling_ogb_reddit_quiver_gpipe.py
```
This script contains old code on experiments using GPipe and traditional DL model parallleism. *Note* that the Quiver sampler can be used instead of the PyG NeighbourSampler, following some type conversions for the adjacency matrix.


### Scalability: Larger Models

```cmd
$ benchmarks/ogbn-papers100M/train_quiver_multi_gpu.py
```
The above is benchmarking and experimental code used for *Single Node Analysis*.


```cmd
$ benchmarks/ogbn-mag240m/train_quiver_multi_node.py
```
The above is benchmarking and experimental code used for *Multi Node Analysis*.



### Report data and diagrams
All end-end benchmark plotting code and data for both sections can be found in the code below. This was mainly used in ** Scalability: Larger Graph Datasets **

```cmd
$ plots/end_to_end/
```
----
