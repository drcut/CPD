# CPD: A High Performance System for Customized-Precision Distributed DL

CPD is a state-of-art system which enables the researchers to use arbirtary precision's floating point 
(with exp <= 8bits, man <= 23bits, as we use IEEE-FP32 floating point to simulate customized-precision floating point) on Pytorch

CPD can support the following operation:

- Cast precision from IEEE-FP32 to customized-precision, and vice versa
- Using customized-precision floating point as accumulator while GEMM operation
- Using customized-precision floating point as mid-result while do all-reduce operation
- Using Kahan accumulation algorithm while accumulation

## Requirements

- System: Ubuntu16.04
- Python >= 3.6
- PyTorch >= 0.4.1
- CUDA >= 9.0
- (Optional) Slurm Workload Manager for the distributed system (If your distrbuted system use other Manager, please implement the function *dist_init* in *CPDtorch/utils/dist_util.py*. This function should assign an unique device for each process and return the global rank and world size for this process)
- Dataset: Cifar10 (for DavidNet and ResNet18) and ImageNet (for ResNet50)


## Get Started
Run ResNet18 with/without APS for 8 bits in a 8 node distributed system

*NOTE: Please modify the platform name ('Test') for your own system*
```bash
# Download code
git clone -b artifact https://github.com/drcut/CPD

# Install CPD
cd CPD/example/ResNet18
ln -s ../../CPDtorch CPDtorch

# Before run the code, please prepare CIFAR10 dataset and put it into data folder, just like below
'''
.
├── configs
│   └── res18_cifar.yaml
├── CPDtorch -> ../../CPDtorch
├── data
    ├── cifar-10-batches-py
    │   ├── batches.meta
    │   ├── data_batch_1
    │   ├── data_batch_2
    │   ├── data_batch_3
    │   ├── data_batch_4
    │   ├── data_batch_5
    │   ├── readme.html
    │   └── test_batch
    └── cifar-10-python.tar.gz
├── models
│   ├── __init__.py
│   └── resnet18_cifar.py
├── tools
│   └── mix.py
└── utils
    ├── __init__.py
    └── train_util.py
'''

# Run ResNet18 for 8 bits (exp: 4bit man: 3bit) without APS
srun -p Test --gres=gpu:8 -n8 --ntasks-per-node=8 python -u tools/mix.py --dist --grad_exp 4 --grad_man 3 | tee no_aps.log

# Run ResNet18 for 8 bits (exp: 4bit man: 3bit) with APS
srun -p Test --gres=gpu:8 -n8 --ntasks-per-node=8 python -u tools/mix.py --dist --grad_exp 4 --grad_man 3 --use_APS | tee aps.log

# (Optional) Visualize the experiments results
python draw_curve.py
```

## Step by Step Instructions
### ResNet18
Following the above command, users can run other experiments with different setting.
```bash
# Run 4 bits (exp: 3bit man: 0bit) with APS
srun -p Test --gres=gpu:8 -n8 --ntasks-per-node=8 python -u tools/mix.py --dist --grad_exp 3 --grad_man 0 --use_APS

# Run 8 bits (exp: 4bit man: 3bit) with APS using LARS algorithm
srun -p Test --gres=gpu:8 -n8 --ntasks-per-node=8 python -u tools/mix.py --dist --grad_exp 4 --grad_man 3 --use_APS --use_lars

# Run 8 bits (exp: 4bit man: 3bit) with APS using the Kahan summation algorithm
srun -p Test --gres=gpu:8 -n8 --ntasks-per-node=8 python -u tools/mix.py --dist --grad_exp 4 --grad_man 3 --use_APS --use_kahan
```
Please feel free to try using different combinations of w/o APS, w/o Kahan, w/o Lars and different precisions to run the experiments

## Acknowledgement
We learned a lot from the following projects when building CPD:

- [QPyTorch](https://github.com/Tiiiger/QPyTorch): We use the same logic for intergrate our work with Pytorch.

## Team
* [Ruobing Han](https://drcut.github.io/)
* [Yang You](https://people.eecs.berkeley.edu/~youyang/)
* [James Demmel](https://people.eecs.berkeley.edu/~demmel/)


