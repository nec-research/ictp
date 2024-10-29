# ICTP: Irreducible Cartesian Tensor Potentials

Official repository for the [paper](https://arxiv.org/abs/2405.14253) _"Higher Rank Irreducible Cartesian Tensors for Equivariant Message Passing"_. It is built upon the [ALEBREW](https://github.com/nec-research/alebrew) repository and implements irreducible Cartesian tensors and their products.

<img src="tensor_and_tensor_product.png" alt="ICTP" width="600"/>

## Citing us

Please consider citing us if you find the code and paper useful:

    @misc{zaverkin2024higherrankirreduciblecartesiantensors,
        title={Higher-Rank Irreducible Cartesian Tensors for Equivariant Message Passing}, 
        author={Viktor Zaverkin and Francesco Alesiani and Takashi Maruyama and Federico Errica and Henrik Christiansen and Makoto Takamoto and Nicolas Weber and Mathias Niepert},
        year={2024},
        eprint={2405.14253},
        archivePrefix={arXiv},
        primaryClass={cs.LG},
        url={https://arxiv.org/abs/2405.14253}, 
    }

## Implemented methods

This repository implements:

- Irreducible Cartesian tensors up to a rank of three;
- Irreducible Cartesian tensor products (currently, only even tensor products);
- MACE-like architecture based on irreducible Cartesian tensors and their products.

## License

This source code has a non-commercial license; see `LICENSE.txt` for more details.

## Requirements

An environment with [PyTorch](https://pytorch.org/get-started/locally/) (>=2.3.1) and [ASE](https://wiki.fysik.dtu.dk/ase/) (==3.22.1) installed.  Also, some other dependencies may be necessary; see the `ictp-cuda.yml` file.

## Installation

First, clone this repository into a directory of your choice `git clone https://github.com/nec-research/ictp.git <dest_dir>`. Then, move to `<dest_dir>` and install the required packages into a conda environment using, e.g., `conda env create -f ictp-cuda.yml`. Finally, set your `PYTHONPATH` environment variable to `export PYTHONPATH=<dest_dir>:$PYTHONPATH`.

## Training potentials with a data set

We provide example scripts for training ICTP models for molecular (`examples/run_training_DHA.py`) and material (`examples/run_training_HEA.py`) systems. For the DHA molecule, first, download the corresponding data set by running `wget http://www.quantum-machine.org/gdml/repo/static/md22_DHA.zip` and unzip it with, e.g., `unzip md22_DHA.zip`. Then, store the `md22_DHA.xyz` file in the `datasets/md22` subfolder and run `python run_training_DHA.py` to train your first ICTP model. The HEA data set can be downloaded from [DaRUS](https://doi.org/10.18419/darus-3516). Please refer to `examples/using_ictp.ipynb` for more details on training ICTP models and using them in, e.g., molecular dynamics simulations.

## How to reproduce the results from the paper

In the `experiments` subfolder, we provide scripts to reproduce all results from the [paper](https://arxiv.org/abs/2405.14253), along with data preparation scripts in the `datasets` subfolder. For the experiments with the original [MACE](https://github.com/ACEsuit/mace) source code, we used the commit `88d49f9ed6925dec07d1777043a36e1fe4872ff3`.