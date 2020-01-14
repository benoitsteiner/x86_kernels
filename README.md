#Sparse model generation
We use the approach from the lottery ticket hypothesis paper to progressively sparsify common models. The corresponding code resides in the models/ directory.

To get started, from the models/ directory, pip install -r requirements.txt to download all the required tools. 

To generate a sparse version of resnet50 (trained on cifar100):
 python3 main.py --prune_type=lt --arch_type=resnet50 --dataset=cifar100 --prune_percent=10 --prune_iterations=35

To generate a sparse version of mobilenet_v2 (trained on cifar100):
 python3 main.py --prune_type=lt --arch_type=mobilenet --dataset=cifar100 --prune_percent=10 --prune_iterations=35

The script will save checkpoints in saves/<model_name>/<dataset>/<iter>_model_lt.pth.tar, where iteration 0 corresponds to the dense model and iteration 35 corresponds to the most sparse version of the model.
The checkpoints can be loaded directly in pytorch by calling torch.load(). .

