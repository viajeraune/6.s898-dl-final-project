from pathlib import Path
import numpy as np
import json
import os
import random
import matplotlib.pyplot as plt

import torch
import torchvision.models as models
import torchvision.transforms as transforms

import src.factories.benchmark_factory as benchmark_factory
import src.factories.method_factory as method_factory
import src.factories.model_factory as model_factory
import src.toolkit.utils as utils

from torch.optim import SGD
from torch.nn import CrossEntropyLoss
from avalanche.training import Naive, Replay, EWC, JointTraining
from avalanche.training.plugins import EarlyStoppingPlugin, ReplayPlugin, EWCPlugin
from avalanche.models import SimpleMLP, SimpleCNN
from avalanche.benchmarks.classic import (SplitCIFAR10, SplitCIFAR100,
                                          SplitImageNet, SplitTinyImageNet)

from avalanche.evaluation.metrics import forgetting_metrics, accuracy_metrics, \
    loss_metrics, timing_metrics, cpu_usage_metrics, confusion_matrix_metrics, disk_usage_metrics
from avalanche.logging import InteractiveLogger, TextLogger, TensorboardLogger
from avalanche.training.plugins import EvaluationPlugin

from avalanche.benchmarks.scenarios import OnlineCLScenario


from avalanche.training.plugins.lr_scheduling import LRSchedulerPlugin
# from avalanche.benchmarks.classic.clear import CLEAR, CLEARMetric # clear

# Define hyperparameters/scheduler/augmentation
HPARAM = {
    "num_experiences": 5,
    "mem_size": 200,
    "train_mb_size": 64, 
    "eval_mb_size": 256,
    "num_epoch": 3,
    "step_scheduler_decay": 30,
    "scheduler_step": 0.1,
    "start_lr": 0.01,
    "weight_decay": 1e-5,
    "momentum": 0.9,
    "num_workers": 1,
}


split_cifar10 = benchmark_factory.create_benchmark("split_cifar10", n_experiences=HPARAM["num_experiences"], val_size=0.05, seed=1, return_task_id=False, class_ids_from_zero_from_first_exp=True)
# split_cifar100 = benchmark_factory.create_benchmark("split_cifar100", n_experiences=HPARAM["num_experiences"], val_size=0.05, seed=1), #n_experiences are same as ocl_survey
# split_miniimagenet = benchmark_factory.create_benchmark("split_miniimagenet", dataset_root=Path("/data"), n_experiences=20), #n_experiences are same as ocl_survey
# split_imagenet = benchmark_factory.create_benchmark("split_imagenet", n_experiences=100), #n_experiences are same as ocl_survey

# scenarios = [split_cifar100, split_miniimagenet, split_imagenet]

# split_cifar10_benchmark = SplitCIFAR10(n_experiences=HPARAM["num_experiences"], return_task_id=True, seed=1)
# print(split_cifar10_benchmark)


DATASET_NAME = "split_cifar10"
NUM_CLASSES = {
    "split_cifar10": 10,
    "split_cifar100": 100,

}
assert DATASET_NAME in NUM_CLASSES.keys()

# Function to extract images from the validation set
def extract_images_from_validation_set(scenario, num_experiences=HPARAM["num_experiences"], images_per_experience=2, seed=0):
    random.seed(seed)  # Set the seed
    selected_images = []
    selected_labels = []

    for experience in scenario.valid_stream[:num_experiences]:
        all_data = list(iter(experience.dataset))

        indices = random.sample(range(len(all_data)), images_per_experience)
        # print(indices)
        
        for idx in indices:
            # print(all_data[idx])
            image, label, _ = all_data[idx]
            selected_images.append(image)
            selected_labels.append(label)

    return selected_images, selected_labels

# Preprocess the image
def preprocess(image, size=32):
    transform = transforms.Compose([
        # transforms.Resize((size, size)),  
        # transforms.ToTensor(),
        transforms.Normalize(
            (0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)
        ),
        transforms.Lambda(lambda x: x[None]),
    ])
    return transform(image)

def deprocess(image):
    image = image.unsqueeze(0)
    transform = transforms.Compose([
        transforms.Lambda(lambda x: x[0]),  # Remove the extra batch dimension if it exists
        transforms.Normalize(mean=[0.0, 0.0, 0.0], std=[1/0.2023, 1/0.1994, 1/0.2010]),  # Inverse of std
        transforms.Normalize(mean=[-0.4914, -0.4822, -0.4465], std=[1.0, 1.0, 1.0]),  # Inverse of mean
        transforms.ToPILImage(),
    ])
    return transform(image)

# Function to generate a saliency map for a given image
def generate_saliency_map(model, image, device):

    # image = preprocess(image).to(device)
    image = image.unsqueeze(0).to(device)
    image.requires_grad_()
    model.eval()

    scores = model(image)
    score_max_index = scores.argmax()
    score_max = scores[0, score_max_index]

    model.zero_grad()
    score_max.backward()

    saliency, _ = torch.max(image.grad.data.abs(), dim=1)
    return saliency[0]

# Function to plot images and their saliency maps
def plot_images_and_saliency_maps(images, saliency_maps):
    assert len(images) == len(saliency_maps)

    fig, axs = plt.subplots(2, len(images), figsize=(20, 4))

    for i in range(len(images)):
        # axs[0, i].imshow(transforms.ToPILImage()(images[i]))  # Original image
        axs[0, i].imshow(deprocess(images[i]))  # Original image
        axs[1, i].imshow(saliency_maps[i].cpu().numpy(), cmap=plt.cm.hot)  # Saliency map
        axs[0, i].axis('off')
        axs[1, i].axis('off')

    plt.show()


def make_scheduler(optimizer, step_size, gamma=0.1):
    scheduler = torch.optim.lr_scheduler.StepLR(
        optimizer, step_size=step_size, gamma=gamma
    )
    return scheduler

# import hydra
# @hydra.main(config_path="config", config_name="config.yaml")

def main():
    # utils.set_seed(config.experiment.seed)
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    model = model_factory.create_model("resnet18", input_size=(32, 32, 3))
    model = model.to(device)

    scenario = split_cifar10
    # log to Tensorboard
    tb_logger = TensorboardLogger()

    # log to text file
    text_logger = TextLogger(open('log.txt', 'w+'))

    # print to stdout
    interactive_logger = InteractiveLogger()

    eval_plugin = EvaluationPlugin(
        accuracy_metrics(minibatch=False, epoch=True, experience=False, stream=True),
        loss_metrics(minibatch=False, epoch=True, experience=False, stream=True),
        timing_metrics(epoch=True, epoch_running=False),
        forgetting_metrics(experience=True, stream=True),
        cpu_usage_metrics(experience=True),
        # confusion_matrix_metrics(num_classes=NUM_CLASSES[DATASET_NAME],  normalize=None, save_image=False,
        #                         stream=True),
        disk_usage_metrics(minibatch=False, epoch=True, experience=True, stream=True),
        loggers=[interactive_logger, text_logger, tb_logger]
    )


    optimizer = torch.optim.SGD(
        model.parameters(),
        # lr=0.0598728574654861,
        lr=HPARAM["start_lr"],
        weight_decay=HPARAM["weight_decay"],
        momentum=HPARAM["momentum"],
    )
 
    scheduler = make_scheduler(
        optimizer,
        HPARAM["step_scheduler_decay"],
        HPARAM["scheduler_step"],
    )

    plugin_list = [LRSchedulerPlugin(scheduler), EarlyStoppingPlugin(patience=10, val_stream_name='train')]

    criterion = torch.nn.CrossEntropyLoss()

    # exp_name = (
    #     config.strategy.name
    #     + "_"
    #     + config.benchmark.factory_args.benchmark_name
    #     + "_"
    #     + str(config.benchmark.factory_args.n_experiences)
    #     + "_"
    #     + str(config.strategy.mem_size)
    # )

    # logdir = os.path.join(
    #     str(config.experiment.results_root),
    #     exp_name,
    #     str(config.experiment.seed),
    # )
    # if config.experiment.logdir is None:
    #     os.makedirs(logdir, exist_ok=True)
    #     utils.clear_tensorboard_files(logdir)

    #     # Add full results dir to config
    #     config.experiment.logdir = logdir
    #     import omegaconf
    #     omegaconf.OmegaConf.save(config, os.path.join(logdir, "config.yaml"))
    # else:
    #     logdir = config.experiment.logdir
        
    # strategy = method_factory.create_strategy(
    #     model=model,
    #     optimizer=optimizer,
    #     plugins=plugin_list,
    #     # logdir=logdir,
    #     name="er",
    #     dataset_name="split_cifar10",
    #     strategy_kwargs=config["strategy"],
    #     evaluation_kwargs=config["evaluation"],
    # )

    strategy = Replay(
        model,
        optimizer,
        criterion=criterion,
        mem_size=HPARAM["mem_size"],
        train_mb_size=HPARAM["train_mb_size"],
        train_epochs=HPARAM["num_epoch"],
        eval_mb_size=HPARAM["eval_mb_size"],
        evaluator=eval_plugin,
        device=device,
        plugins=plugin_list,
        eval_every=0 # evaluation frequency. If 0, evaluate after each experience
    )

    # strategy = EWC(
    #     model=model,
    #     optimizer=optimizer,
    #     criterion=criterion,
    #     train_mb_size=HPARAM["train_mb_size"],
    #     train_epochs=3,
    #     eval_mb_size=HPARAM["eval_mb_size"],
    #     device=device,
    #     evaluator=eval_plugin,
    #     plugins=plugin_list,
    #     ewc_lambda=0.4,
    # )

    # strategy = Naive(
    #     model,
    #     optimizer,
    #     criterion=criterion,
    #     train_mb_size=HPARAM["train_mb_size"],
    #     train_epochs=HPARAM["num_epoch"],
    #     eval_mb_size=HPARAM["eval_mb_size"],
    #     evaluator=eval_plugin,
    #     device=device,
    #     plugins=plugin_list,
    # )

    print("Using strategy: ", strategy.__class__.__name__)
    print("With plugins: ", strategy.plugins)

    is_online = True

    # Training loop
    # For CIFAR100, need to indicate [0] to each scenario instance to get the first experience
    batch_streams = scenario.streams.values()
    for t, experience in enumerate(scenario.train_stream):
        if is_online == True:
            print("Online CL")
            ocl_scenario = OnlineCLScenario(
                original_streams=batch_streams,
                experiences=experience,
                experience_size=HPARAM["train_mb_size"],
                access_task_boundaries=False,
                shuffle=True,
            )
            train_stream = ocl_scenario.train_stream
        else:
            train_stream = experience

        strategy.train(
            train_stream,
            # eval_streams=[scenario.valid_stream[: t + 1]],
            num_workers=HPARAM["num_workers"],
            drop_last=True,
        )

        # torch.save(
        #     strategy.model.state_dict(), os.path.join(logdir, f"model_{t}.ckpt")
        # )

        results = strategy.eval(scenario.test_stream[: t + 1])
    # results = strategy.eval(scenario.test_stream)

    # Extract images from the validation set
    selected_images, selected_labels = extract_images_from_validation_set(scenario)

    # Generate saliency maps for the selected images
    saliency_maps = []
    for image in selected_images:
        saliency_map = generate_saliency_map(model, image, device)
        saliency_maps.append(saliency_map)

    # Plot the images and their saliency maps
    plot_images_and_saliency_maps(selected_images, saliency_maps)

    return model, results


if __name__ == "__main__":
   model, results = main()
