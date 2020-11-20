# CPD: A High Performance System for Customized-Precision Distributed DL

CPD is a state-of-art system which enables the researchers to use arbirtary precision's floating point 
(with exp <= 8bits, man <= 23bits, as we use IEEE-FP32 floating point to simulate customized-precision floating point) on Pytorch

CPD can support the following operation:

- Cast precision from IEEE-FP32 to customized-precision, and vice versa
- Using customized-precision floating point as accumulator while GEMM operation
- Using customized-precision floating point as mid-result while do all-reduce operation
- Using Kahan accumulation algorithm while accumulation

*Note*: Besides customized-precision, we also implement a **high performance general** GEMM function for CUDA,
as we noticed there isn't an open source GEMM implemention which are high performance for general case (with random M, N and K),
except cuTLASS, which are too complex to modify.
 Researchers can use them with IEEE single floating points.



## Requirements

- System: Ubuntu16.04
- Python >= 3.6
- PyTorch >= 0.4.1
- CUDA >= 9.0
- MMCV >= 1.1.3 (Only for segmentation model, like FCN)
- MMSegmentation >= 0.6.0 (Only for segmentation model, like FCN)
- (Optional) Slurm Workload Manager for the distributed system (If your distrbuted system use other Manager, please implement the function *dist_init* in *CPDtorch/utils/dist_util.py*. This function should assign an unique device for each process and return the global rank and world size for this process)
- Dataset: Cifar10 (for DavidNet and ResNet18) and ImageNet (for ResNet50)


## Get Started
Run ResNet18 with/without APS for 8 bits in a 8 node distributed system

*NOTE: Please modify the platform name ('Test') for your own system*
```bash
# Download code
git clone -b artifact https://github.com/drcut/CPD
# Install CPD
export PYTHONPATH=`pwd`/CPD:$PYTHONPATH

# Run ResNet18
cd CPD/example/ResNet18
# Before run the code, please prepare CIFAR10 dataset and put it into data folder, just like below
'''
.
├── configs
│   └── res18_cifar.yaml
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

### DavidNet
```bash
cd CPD/example/DavidNet
# use the same way to install CPDtorch and prepare CIFAR10 dataset.Your folder should like the below case
'''
.      
├── data -> ../ResNet18/data                                            
├── davidnet.py                                                         
├── dawn.py                                        
├── train_utils.py                                                      
├── use_aps.log                                                          
└── utils.py
'''
# Run 8 bits (exp: 5bit man: 2bit) with APS
srun -p Test --gres=gpu:8 -n8 --ntasks-per-node=8 python -u dawn.py --grad_exp 5 --grad_man 2 --use_APS
```

### ResNet50
*For an 8 V100 distrbuted system, it may take more than 30 hour for 90 epochs training*
```bash
# Install CPD
cd CPD/example/ResNet50

# Prepare ImageNet data, you should modify args.data in main.py

# Run ResNet50 with 8 bit (exp: 5bit, man: 2bit) with APS in a 8 node distributed system
srun -p Test --gres=gpu:8 -n8 --ntasks-per-node=8 python -u main.py --grad_exp 5 --grad_man 2 --use-APS

# If you only have 8 node, but want to emulate the experiments for a 256 node system. That means you should use a node to emulate 256/8=32 nodes. So set emulate-node as 32
srun -p Test --gres=gpu:8 -n8 --ntasks-per-node=8 python -u main.py --grad_exp 5 --grad_man 2 --use-APS --emulate-node 32
```

### FCN
*For an 8 V100 distributed system, it may take more than 20 hour for 40K iteration*
```bash
# Donwload MMCV for APS
git clone -b APS_support https://github.com/drcut/mmcv
# Build MMCV following the official link: 
# https://github.com/open-mmlab/mmcv#installation
cd mmcv
MMCV_WITH_OPS=1 pip install --user -e .
# Build MMSeg following the official link: 
# https://github.com/open-mmlab/mmsegmentation/blob/master/docs/install.md
cd ..
git clone -b v0.5.0 https://github.com/open-mmlab/mmsegmentation
cd mmsegmentation
pip install --user -e .

# Run FCN according to instructions in : https://github.com/open-mmlab/mmsegmentation/blob/master/docs/getting_started.md#train-with-multiple-gpus
# Please modify the code in line 27th mmcv/runner/hooks/optimizer.py for different precision and open/close APS
GPUS=8 GPUS_PER_NODE=8 CPUS_PER_TASK=1 bash tools/slurm_train.sh Test no_APS_4_3 configs/fcn/fcn_r50-d8_769x769_40k_cityscapes.py
```

## Claims from the paper supported by the artifact
In our artifact, we can verify that using APS, we can improve the testing accuracies when distributed training with low precision gradient in several models (DavidNet, ResNet18, FCN, ResNet50). Users can further test other models easily using our framework.

##  Claims from the paper not supported by the artifact
As CPD is an emulator for low precision training, we actual use IEEE FP32 in out training. So this means we can not show the performance improvement in this artifact. However, we are designing a new computer architecture with a startup company for APS. With this kind of architecture, we can gain speedup by using APS with low precision.


## Acknowledgement
We learned a lot from the following projects when building CPD:

- [QPyTorch](https://github.com/Tiiiger/QPyTorch): We use the same logic for intergrate our work with Pytorch.

## Team
* [Ruobing Han](https://drcut.github.io/)
* [Yang You](https://people.eecs.berkeley.edu/~youyang/)
* [James Demmel](https://people.eecs.berkeley.edu/~demmel/)


