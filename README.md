# DL Framework in PyTorch

We build a mini-deep learning framework to train simple feedforward networks. The framework is built using the standard tensor operations in PyTorch. It supports the following modules:
- Linear
- ReLU
- Tanh
- LossMSE
- Sequential


### Experiments

We provide a toy example to demonstrate our framework. The following commands reproduce the results provided in the report. 

1. To run the model for 1 run:
```
python test.py
```

2. To run the model for 10 runs:
```
python test.py --num_runs 10
```

3. To run the PyTorch model for 10 runs:
```
python test_pytorchnn.py --num_runs 10
```

4. To perform the ablation on batch size:
```
make batch_size_ablation
```
NOTE: If 'make' command is not allowed for the project, you can run the commands individually that are provided in the Makefile.
