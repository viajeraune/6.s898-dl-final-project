import torch
import torchvision
from torch.optim import SGD
from torch.nn import CrossEntropyLoss
from avalanche.benchmarks.classic import SplitMNIST, SplitCIFAR10, SplitInaturalist
from avalanche.training import Naive, Replay, EWC, JointTraining
from avalanche.models import SimpleMLP, SimpleCNN
from avalanche.benchmarks.scenarios import OnlineCLScenario


device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")


scenario = SplitMNIST(n_experiences=5, seed=1)

model = SimpleMLP(num_classes=scenario.n_classes)

cl_strategy_naive = Naive(
    model,
    optimizer=SGD(model.parameters(), lr=0.001, momentum=0.9),
    criterion=CrossEntropyLoss(),
    train_mb_size=100,
    train_epochs=4,
    eval_mb_size=100,
    device=device
    # evaluator=eval_plugin
)

def experiment(scenario, strategy, is_online):
  print("Using strategy: ", strategy.__class__.__name__)
  print("With plugins: ", strategy.plugins)

  batch_streams = scenario.streams.values()
  for t, experience in enumerate(scenario.train_stream):
    if is_online == True:
        ocl_scenario = OnlineCLScenario(
            original_streams=batch_streams,
            experiences=experience,
            experience_size=1,
            # access_task_boundaries=False,
        )
        train_stream = ocl_scenario.train_stream
    else:
        train_stream = experience

    strategy.train(
        train_stream,
        # eval_streams=[scenario.valid_stream[: t + 1]],
        num_workers=0,
        # drop_last=True,
    )

    strategy.eval(scenario.test_stream[t]) # for current test set
    results = strategy.eval(scenario.test_stream) # for entire test set
  return results

experiment(scenario, cl_strategy_naive, False)