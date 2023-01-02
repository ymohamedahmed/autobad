# autobad

Simple implementation of reverse-mode, vectorized autodiff. Only dependencies are `numpy`, `more_itertools` and `graphviz` (for visualising the computation graph). 

It is an eager implementation. As computations are completed, appropriate edges are added to the backwards graph (i.e. like PyTorch) rather than the user building a graph upfront.

For an example MLP see `examples/mlp.py`. 