Sparse model generation
-----------------------
We use the approach from the lottery ticket hypothesis paper to progressively sparsify common models. The corresponding code resides in the models/ directory.

To get started, from the models/ directory, pip3 install -r requirements.txt to download all the required tools. Then uninstall the typing package since pytorch isn't compatible with it: pip3 uninstall typing 

To generate a sparse version of resnet50 (trained on cifar100):
 python3 main.py --prune_type=lt --arch_type=resnet50 --dataset=cifar100 --prune_percent=10 --prune_iterations=35

To generate a sparse version of mobilenet_v2 (trained on cifar100):
 python3 main.py --prune_type=lt --arch_type=mobilenet --dataset=cifar100 --prune_percent=10 --prune_iterations=35

To generate a sparse version of the transformer model:
 python3 main.py --prune_type=lt --arch_type=transformer --dataset=iwslt14 --prune_percent=10 --prune_iterations=35 --end_iter=1

NCF:
 python3 main.py --prune_type=lt --arch_type=neumf --dataset=ncf --prune_percent=10 --prune_iterations=35  --batch_size=256 --end_iter=1

The script will save checkpoints in saves/<model_name>/<dataset>/<iter>_model_lt.pth.tar, where iteration 0 corresponds to the dense model and iteration 35 corresponds to the most sparse version of the model.
The checkpoints can be loaded directly in pytorch by calling torch.load(). .

