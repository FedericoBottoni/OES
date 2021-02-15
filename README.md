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

Otherwise select BL_TT to run the one-brain baseline 


## Configure and run the experiments


```bash
python main.py
```

## Author
<a href="https://github.com/FedericoBottoni/OES/graphs/contributors">
  <img src="https://contributors-img.web.app/image?repo=FedericoBottoni/OES" />
</a>

* **Federico Bottoni** - [FedericoBottoni](https://github.com/federicobottoni)
