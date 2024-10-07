from avalanche.models import SimpleMLP

from avalanche.benchmarks.classic import SplitMNIST

from avalanche.logging import InteractiveLogger, TextLogger, TensorboardLogger

from avalanche.training.plugins import EvaluationPlugin
from avalanche.evaluation.metrics import forgetting_metrics, accuracy_metrics, loss_metrics

from avalanche.training import Replay
from torch.optim import SGD
from torch.nn import CrossEntropyLoss
import torch

import os
import matplotlib.pyplot as plt
from tensorboard.backend.event_processing.event_accumulator import EventAccumulator 

def plot_tensorboard_logs(log_dir, tag, xlabel='Iterations', ylabel='Value', title='TensorBoard Log'):
    """
    Plots a specific tag from a TensorBoard log directory.

    :param log_dir: Directory where TensorBoard log files are stored.
    :param tag: The tag to plot (e.g., 'Accuracy_Task1/Every_100_Iterations').
    :param xlabel: Label for the x-axis (default: 'Iterations').
    :param ylabel: Label for the y-axis (default: 'Value').
    :param title: Title of the plot (default: 'TensorBoard Log').
    """
    event_acc = EventAccumulator(log_dir)
    event_acc.Reload()

    # Extract data for the specific tag
    tags = event_acc.Tags()["scalars"]
    if tag not in tags:
        print(f"Tag '{tag}' not found in TensorBoard logs.")
        return

    events = event_acc.Scalars(tag)

    # Get iterations and values
    iterations = [e.step for e in events]
    values = [e.value for e in events]

    # Plot
    plt.figure(figsize=(10, 6))
    plt.plot(iterations, values, label=tag)
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    plt.title(title)
    plt.legend()
    plt.grid(True)
    plt.show()

class MyReplay(Replay):
    def __init__(self, *args, tb_writer=None, **kwargs):
        super().__init__(*args, **kwargs)
        self.tb_writer = tb_writer  # TensorBoard writer to log task-specific metrics
        self.iter_count = 0  # Initialize iteration counter
    
        
    def _after_training_iteration(self, **kwargs):
        super()._after_training_iteration(**kwargs)
        self.iter_count = (self.iter_count + 1)

        results = self.eval(benchmark.test_stream[0])
        if 'Top1_Acc_Exp/eval_phase/test_stream/Task000/Exp000' in results:
            acc_task1 = results['Top1_Acc_Exp/eval_phase/test_stream/Task000/Exp000']
            self.tb_writer.add_scalar("Accuracy with momentum=0.1", acc_task1, self.iter_count)


        if self.iter_count%100 == 0 and 'Top1_Acc_Exp/eval_phase/test_stream/Task000/Exp000' in results:
            acc_task1 = results['Top1_Acc_Exp/eval_phase/test_stream/Task000/Exp000']
            self.tb_writer.add_scalar("Accuracy_Task1/Every_100_Iterations", acc_task1, self.iter_count)



#benchmark
benchmark = SplitMNIST(n_experiences = 5, seed=123, return_task_id=True)

#model
model = SimpleMLP(num_classes=benchmark.n_classes, hidden_layers=2, hidden_size=400, drop_rate=0)

#logger
tb_logger = TensorboardLogger(tb_log_dir="./tensorboard_logs/m0.1")
interactive_logger = InteractiveLogger()

#eval 
eval_plugin = EvaluationPlugin(
    accuracy_metrics(minibatch=True, epoch=True, experience=True, stream=True),
    loss_metrics(minibatch=True, epoch=True, experience=True, stream=True),
    forgetting_metrics(experience=True, stream=True),
    loggers=[interactive_logger, tb_logger]
)

#strategy
#lr appartiene a {0.1, 0.01, 0.001, 0.0001}
#bach size = 256
# α ∈ {0.1, 0.3, 0.5, 0.7, 0.9}
#             (learning rate, loss weight, buffer)
#Split-MNIST (0.01, 0.3, 2 · 10^3)


strategy = MyReplay(
    model=model,
    optimizer = SGD(model.parameters(), lr=0.01, momentum=0.1),
    criterion = CrossEntropyLoss(),
    mem_size = 2000,
    train_mb_size = 256,
    eval_mb_size = 256,
    train_epochs = 10,
    evaluator = eval_plugin,
    eval_every = 1,
    tb_writer = tb_logger.writer
)

print("Experiment Start")

for experience in benchmark.train_stream:    
    res = strategy.train(experience)


all_metrics = eval_plugin.get_all_metrics() 
print(all_metrics['Top1_Acc_Exp/eval_phase/test_stream/Task000/Exp000'][1])
