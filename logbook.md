Some random notes whilst developing.

- Need to be able to average gradients between samples in a batch, but sum gradients if a parameter is re-used within a sample. Option 1: call backward on each sample and find a way to tell `backward` to average the gradients. Or call backward on the average loss
- It's a bit annoying to write a general vjp that also deals with no incoming gradient.
- Graph representation: tensor-first seemed much easier to implement. Alternative scheme involved both tensor nodes and function nodes, but felt more complex to implement.