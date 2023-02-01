# Federated Learning as Variational Inference: A Scalable Expectation Propagation Approach

This repo contains the code for paper Federated Learning as Variational Inference: A Scalable Expectation Propagation Approach, and is built on top of [FedPA](https://github.com/alshedivat/fedpa).

## Setup
```bash
bash setup.sh
```

## Experiments

1. First run the following command to get the base model checkpoints.
```bash
python run_experiments.py -m gpus=[0] tasks=cifar100 which_experiment=baseline base_file_name=baseline
```

2. Put the path to the saved model checkpoints in this [file](./federated/experiments/file_utils.py).

3. Use the following commands to run the `Fed[S]EP` experiments.
```bash
# FedSEP Scaled Identity
python run_experiments.py -m gpus=[0] tasks=cifar100 moment.method=identity moment.scale=500.0
# FedSEP MCMC
python run_experiments.py -m gpus=[0] tasks=cifar100 moment.method=mcmc     moment.scale=500.0 moment.shrinkage=1e-4
# FedSEP Laplce
python run_experiments.py -m gpus=[0] tasks=cifar100 moment.method=laplace  moment.scale=500.0 moment.num_epochs=5
# FedSEP NGVI
python run_experiments.py -m gpus=[0] tasks=cifar100 moment.method=vi       moment.scale=500.0 moment.num_epochs=5 moment.Lambda_decay_rate=0.99 moment.num_samples=5

# FedEP Scaled Identity
python run_experiments.py -m gpus=[0] tasks=cifar100 algorithm_name=ep moment.method=identity moment.scale=500.0 tasks.optim_max_norm=100.0
# FedEP MCMC
python run_experiments.py -m gpus=[0] tasks=cifar100 algorithm_name=ep moment.method=mcmc     moment.scale=500.0 tasks.optim_max_norm=300.0 moment.shrinkage=1e-4
# FedEP Laplce
python run_experiments.py -m gpus=[0] tasks=cifar100 algorithm_name=ep moment.method=laplace  moment.scale=500.0 tasks.optim_max_norm=6500.0 moment.num_epochs=10
# FedEP NGVI
python run_experiments.py -m gpus=[0] tasks=cifar100 algorithm_name=ep moment.method=vi       moment.scale=500.0 tasks.optim_max_norm=2000.0 moment.num_epochs=5 moment.Lambda_decay_rate=0.99 moment.num_samples=5
```

## Citation
```bibtex
@inproceedings{alshedivat2021federated,
  title={Federated Learning via Posterior Averaging: A New Perspective and Practical Algorithms},
  author={Al-Shedivat, Maruan and Gillenwater, Jennifer and Xing, Eric and Rostamizadeh, Afshin},
  booktitle={International Conference on Learning Representations (ICLR)},
  year={2021}
}
```

## License

Apache 2.0
