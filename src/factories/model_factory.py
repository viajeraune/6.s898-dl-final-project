import torch
import torchvision.models as tvmodels

import avalanche.models as models
import src.toolkit.utils as utils
from avalanche.models.dynamic_modules import IncrementalClassifier
from avalanche.training.plugins import LRSchedulerPlugin
from src.toolkit.slim_resnet18 import SlimResNet18


# def get_last_layer_name(model):
#     last_layer_name = list(model.named_parameters())[-1][0].split(".")[0]

#     last_layer = getattr(model, last_layer_name)

#     if isinstance(last_layer, IncrementalClassifier):
#         in_features = last_layer.classifier.in_features
#     else:
#         in_features = last_layer.in_features
#     return last_layer_name, in_features

def create_model(model_type: str, input_size):
    if model_type == "slim_resnet18":
        model = SlimResNet18(1, input_size=input_size)
    if model_type == "resnet18":
        model = tvmodels.resnet18()
    if model_type == "resnet34":
        model = tvmodels.resnet34()
    if model_type == "resnet50":
        model = tvmodels.resnet50()
    if model_type == "alexnet":
        model = tvmodels.alexnet()

    # last_layer_name, in_features = utils.get_last_layer_name(model)
    last_layer_name, in_features = utils.get_last_layer_name(model)


    classifier = IncrementalClassifier(in_features, 1)

    setattr(model, last_layer_name, classifier)

    return model


def get_optimizer(
    model,
    optimizer_type: str,
    kwargs_optimizer=None,
    scheduler_type: str = None,
    kwargs_scheduler=None,
):
    if optimizer_type == "SGD":
        sgd_args = ["lr", "momentum", "weight_decay"]
        sgd_args = utils.extract_kwargs(sgd_args, kwargs_optimizer)
        optimizer = torch.optim.SGD(model.parameters(), **sgd_args)
    elif optimizer_type == "Adam":
        adam_args = ["lr", "weight_decay"]
        adam_args = utils.extract_kwargs(adam_args, kwargs_optimizer)
        optimizer = torch.optim.Adam(model.parameters(), **adam_args)
    elif optimizer_type == "AdamW":
        adam_args = ["lr", "weight_decay"]
        adam_args = utils.extract_kwargs(adam_args, kwargs_optimizer)
        optimizer = torch.optim.AdamW(model.parameters(), **adam_args)

    assert optimizer is not None

    # Scheduler
    scheduler = None
    if scheduler_type == "step":
        assert kwargs_scheduler is not None
        scheduler_args = ["gamma", "step_size"]
        scheduler_args = utils.extract_kwargs(scheduler_args, kwargs_scheduler)
        scheduler = torch.optim.lr_scheduler.StepLR(optimizer, **scheduler_args)

    if scheduler_type == "cosine":
        assert kwargs_scheduler is not None
        scheduler_args = ["eta_min", "T_max"]
        scheduler_args = utils.extract_kwargs(scheduler_args, kwargs_scheduler)
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            optimizer, **scheduler_args
        )

    if scheduler is not None:
        plugin_args = ["reset_scheduler", "reset_lr", "step_granularity"]
        plugin_args = utils.extract_kwargs(plugin_args, kwargs_scheduler)
        scheduler_plugin = LRSchedulerPlugin(
            scheduler, **plugin_args
        )
    else:
        scheduler_plugin = None

    return optimizer, scheduler_plugin
