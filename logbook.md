Some random notes whilst developing.

- Need to be able to average gradients between samples in a batch, but sum gradients if a parameter is re-used within a sample.
- Graph representation: tensor-first seemed much easier to implement. Alternative scheme involved both tensor nodes and function nodes, but felt more complex to implement.