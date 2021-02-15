# Online Experience Sharing (OES)

> This project consists in Online Experience Sharing, a framework establishing communications among Deep Reinforcement Learning agents to highlight the Transfer Learning improvements brought by the experience to transfer selection methods, state confidence systems and transfer frequencies and sizes. Details available in the related document.

## Requirements

* OpenAI Gym
* NumPy
* PyTorch
* Tensorboard
* Matplotlib

## Installation
You can install the dependencies using pip
```bash
git clone https://github.com/FedericoBottoni/OES
cd OES
pip install -r requirements.txt
```

## Select the right branch
Select the scenario to test among the available branches:
* State Visit Table (VT)
* State Random Network Distillation (S-RND)
* State-Action Random Network Distillation (Q-RND)

Otherwise select "BL_TT" to run the one-brain baseline or "VT-AutoML" to run parameters tuning through Bayesian Optimisation.

## Configure and run the experiments
Customize the called functions from "provide_transfer" (for SSM) and "gather_transfer" for (RSM) in the related transfer module:
* VT: transfer/visits_filters.py
* SRND: transfer/rnd_filters.py
* QRND: transfer/q_rnd_filters.py

Successively run the main entry point:

```bash
python main.py
```

## Run Bayesian Optimisation
Select "VT-AutoML" branch, configure auto_config.json for parameters' ranges and run the following entry point:

```bash
python hp_sampling.py
```

## Author
<a href="https://github.com/FedericoBottoni/OES/graphs/contributors">
  <img src="https://contributors-img.web.app/image?repo=FedericoBottoni/OES" />
</a>

* **Federico Bottoni** - [FedericoBottoni](https://github.com/federicobottoni)
