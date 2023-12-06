from pathlib import Path
import numpy as np
import json

import torch
import torchvision
import src.factories.benchmark_factory as benchmark_factory
import src.factories.method_factory as method_factory
import src.factories.model_factory as model_factory

from avalanche.benchmarks.classic import (SplitCIFAR10, SplitCIFAR100,
                                          SplitImageNet, SplitTinyImageNet, SplitMNIST)

from torch.optim import SGD
from torch.nn import CrossEntropyLoss
from avalanche.benchmarks.classic import SplitMNIST, SplitCIFAR10, SplitInaturalist
from avalanche.training import Naive, Replay, EWC, JointTraining
from avalanche.models import SimpleMLP, SimpleCNN

from avalanche.evaluation.metrics import forgetting_metrics, accuracy_metrics, \
    loss_metrics, timing_metrics, cpu_usage_metrics, confusion_matrix_metrics, disk_usage_metrics
from avalanche.logging import InteractiveLogger, TextLogger, TensorboardLogger
from avalanche.training.plugins import EvaluationPlugin

from avalanche.benchmarks.scenarios import OnlineCLScenario


from avalanche.training.plugins.lr_scheduling import LRSchedulerPlugin
# from avalanche.benchmarks.classic.clear import CLEAR, CLEARMetric # clear




# # please refer to paper for discussion on streaming v.s. iid protocol
# EVALUATION_PROTOCOL = "streaming"  # trainset = testset per timestamp
# # EVALUATION_PROTOCOL = "iid"  # 7:3 trainset_size:testset_size

# # Define the paths
# ROOT = Path("..")
# DATA_ROOT = ROOT / "data" / DATASET_NAME
# MODEL_ROOT = ROOT / "models"
# # ZIP_FILE = DATA_ROOT / "clear10-train.zip"  # Path to the zip file

# # Create directories if they don't exist
# DATA_ROOT.mkdir(parents=True, exist_ok=True)
# MODEL_ROOT.mkdir(parents=True, exist_ok=True)

# Check if the dataset is already extracted
# if not (DATA_ROOT / "train").exists():  # Replace with actual folder name inside zip
#     # Extract the dataset
#     with zipfile.ZipFile(ZIP_FILE, 'r') as zip_ref:
#         zip_ref.extractall(DATA_ROOT)

# Define hyperparameters/scheduler/augmentation
HPARAM = {
    "batch_size": 256,
    "num_epoch": 10,
    "step_scheduler_decay": 30,
    "scheduler_step": 0.1,
    "start_lr": 0.01,
    "weight_decay": 1e-5,
    "momentum": 0.9,
}


split_cifar10 = benchmark_factory.create_benchmark("split_cifar10", n_experiences=5, val_size=0.2, seed=1)
# split_cifar100 = benchmark_factory.create_benchmark("split_cifar100", n_experiences=20, val_size=0.2, seed=1), #n_experiences are same as ocl_survey
# split_miniimagenet = benchmark_factory.create_benchmark("split_miniimagenet", dataset_root=Path("/data"), n_experiences=20), #n_experiences are same as ocl_survey
# split_imagenet = benchmark_factory.create_benchmark("split_imagenet", n_experiences=100), #n_experiences are same as ocl_survey

# scenarios = [split_cifar100, split_miniimagenet, split_imagenet]


DATASET_NAME = "split_cifar10"
NUM_CLASSES = {
    "split_cifar10": 10,
    "split_cifar100": 100,

}
assert DATASET_NAME in NUM_CLASSES.keys()


def make_scheduler(optimizer, step_size, gamma=0.1):
    scheduler = torch.optim.lr_scheduler.StepLR(
        optimizer, step_size=step_size, gamma=gamma
    )
    return scheduler

def main():
    model = model_factory.create_model("resnet18", input_size=(32, 32, 3))

    scenario = split_cifar10

    # log to Tensorboard
    tb_logger = TensorboardLogger()

    # log to text file
    text_logger = TextLogger(open('log.txt', 'a'))

    # print to stdout
    interactive_logger = InteractiveLogger()

    eval_plugin = EvaluationPlugin(
        accuracy_metrics(minibatch=True, epoch=True, experience=True, stream=True),
        loss_metrics(minibatch=True, epoch=True, experience=True, stream=True),
        timing_metrics(epoch=True, epoch_running=True),
        forgetting_metrics(experience=True, stream=True),
        cpu_usage_metrics(experience=True),
        confusion_matrix_metrics(num_classes=NUM_CLASSES[DATASET_NAME],  normalize=None, save_image=False,
                                stream=True),
        disk_usage_metrics(minibatch=True, epoch=True, experience=True, stream=True),
        loggers=[interactive_logger, text_logger, tb_logger]
    )


    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    model = model.to(device)

    optimizer = torch.optim.SGD(
        model.parameters(),
        lr=0.0598728574654861,
        # lr=HPARAM["start_lr"],
        # weight_decay=HPARAM["weight_decay"],
        # momentum=HPARAM["momentum"],
    )
 
    scheduler = make_scheduler(
        optimizer,
        HPARAM["step_scheduler_decay"],
        HPARAM["scheduler_step"],
    )

    plugin_list = [LRSchedulerPlugin(scheduler)]

    # strategy = method_factory.create_strategy(
    #     model=model,
    #     optimizer=optimizer,
    #     plugins=plugin_list,
    #     # logdir=logdir,
    #     name="er",
    #     # dataset_name=config.benchmark.factory_args.benchmark_name,
    #     # strategy_kwargs=config["strategy"],
    #     # evaluation_kwargs=config["evaluation"],
    # )


    strategy = Naive(
        model,
        optimizer,
        torch.nn.CrossEntropyLoss(),
        train_mb_size=HPARAM["batch_size"],
        train_epochs=HPARAM["num_epoch"],
        eval_mb_size=HPARAM["batch_size"],
        evaluator=eval_plugin,
        device=device,
        plugins=plugin_list,
    )

    print("Using strategy: ", strategy.__class__.__name__)
    print("With plugins: ", strategy.plugins)

    is_online = False

    # Training loop
    # For CIFAR100, need to indicate [0] to each scenario instance to get the first experience
    batch_streams = scenario.streams.values()
    for t, experience in enumerate(scenario.train_stream):
        if is_online == True:
            ocl_scenario = OnlineCLScenario(
                original_streams=batch_streams,
                experiences=experience,
                experience_size=1,
                access_task_boundaries=False,
            )
            train_stream = ocl_scenario.train_stream
        else:
            train_stream = experience

        strategy.train(
            train_stream,
            eval_streams=[scenario.valid_stream[: t + 1]],
            num_workers=0,
            drop_last=True,
        )

        results = strategy.eval(scenario.test_stream[: t + 1])

    return results


if __name__ == "__main__":
    main()

