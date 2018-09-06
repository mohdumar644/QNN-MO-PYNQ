# Training & Packing Weights

Install **tensorpack** on top of TensorFlow.

Run the ``mnist_dorefa.py`` script to train.

Export the weights using

```
tensorpack/scripts/dump-model-params.py   --meta   train_log/PATH-TO-GRAPH-META-FILE   path-to-CheckPoint-file   mnist.npz
```

Then run ``bin-gen.py`` to pack the weights into .bin files.

Finally copy the ``conv0``, ``fc0`` & ``fc1`` folders from the .npz archive into the newly created ``binparam-mnist`` folder.
