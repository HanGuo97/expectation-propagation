import os
import wandb
import hydra
from omegaconf import OmegaConf, DictConfig, ListConfig
from hydra.core.hydra_config import HydraConfig

from federated.experiments.main import run_experiment
os.environ["WANDB_START_METHOD"] = "thread"


@hydra.main(config_path="federated/configs", config_name="config")
def main(config: DictConfig) -> None:
    if not isinstance(config.gpus, ListConfig):
        raise TypeError(f"GPUs must be a `ListConfig`, but {type(config.gpus)}")
    jobid = HydraConfig.get().job.num
    gpuid = config.gpus[jobid % len(config.gpus)]

    # https://stackoverflow.com/questions/37893755/tensorflow-set-cuda-visible-devices-within-jupyter
    os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"   # see issue #152
    os.environ["CUDA_VISIBLE_DEVICES"] = f"{gpuid}"

    # Weight and Bias Logging
    wandb_run = wandb.init(
        config=OmegaConf.to_container(config),
        project="federated",
        reinit=True)

    print(f"Running Job #{jobid} on GPU #{gpuid}")
    print(OmegaConf.to_yaml(config))
    with wandb_run:
        run_experiment(config)


if __name__ == "__main__":
    main()
