# Dichotomize and Generalize: PAC-Bayesian Binary Activated Deep Neural Networks

This repository contains an implementation of PBGNet (**P**AC-Bayesian **B**inary **G**radient **Net**work) and all related experiments presented in "[Dichotomize and Generalize: PAC-Bayesian Binary Activated Deep Neural Networks](https://papers.nips.cc/paper/8911-dichotomize-and-generalize-pac-bayesian-binary-activated-deep-neural-networks)" by Letarte, Germain, Guedj and Laviolette, accepted at *NeurIPS 2019*.

## Requirements
- Python 3.6
- Numpy 1.14.3
- Pytorch 1.2.0
- Poutyne 1.2
- Scikit-learn 0.20.3
- Pandas 0.23.0
- Click 6.7

## Launching
To reproduce the experiment presented in Section 6 of the paper, run:
```zsh
python launch.py
```
To launch a single learning experiment with custom options, use ``experiment.py``.
Here is an example:
```zsh
python experiment.py -d mnist17 -n pbgnet --experiment-name my_exp --sample-size 50 --hidden-size 25
```
For all possible options and their description, see ``python experiment.py --help``.

## BiBTeX
```
@inproceedings{letarte2019dichotomize,
  title={Dichotomize and generalize: Pac-bayesian binary activated deep neural networks},
  author={Letarte, Ga{\"e}l and Germain, Pascal and Guedj, Benjamin and Laviolette, Fran{\c{c}}ois},
  booktitle={Advances in Neural Information Processing Systems},
  pages={6869--6879},
  year={2019}
}
```
