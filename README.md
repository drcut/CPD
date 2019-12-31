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

## Installation

requirements:

- Python >= 3.6
- PyTorch >= 0.4.1


Install CPD is quite easy, just link the CPDtorch file with your project:
```bash
ln -s $PATH_of_this_repo/CPDtorch $PATH_of_your_project/CPDtorch
```


## Examples
We use our large batch low-precision experiment's code as example. 

- [Distributed Low Precision examples](https://github.com/drcut/customized_precision_test)


## Acknowledgement
We learned a lot from the following projects when building CPD:

- [QPyTorch](https://github.com/Tiiiger/QPyTorch): We use the same logic for intergrate our work with Pytorch.

## Team
* [Ruobing Han](https://drcut.github.io/)
* [Yang You](https://people.eecs.berkeley.edu/~youyang/)
* [James Demmel](https://people.eecs.berkeley.edu/~demmel/)

