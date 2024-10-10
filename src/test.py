from decimal import Decimal
from avalanche.models import SimpleMLP

from avalanche.benchmarks.classic import SplitMNIST

from avalanche.logging import InteractiveLogger, TextLogger, TensorboardLogger

from avalanche.training.plugins import EvaluationPlugin
from avalanche.evaluation.metrics import forgetting_metrics, accuracy_metrics, loss_metrics

from avalanche.training import Replay
from torch.optim import SGD
from torch.nn import CrossEntropyLoss
import torch

import sys


#strategy plugin
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
            self.tb_writer.add_scalar("Accuracy", acc_task1, self.iter_count)



n_seeds = int(sys.argv[2])
momentum = sys.argv[1]

for i in range(0, n_seeds):
    benchmark = SplitMNIST(n_experiences = 5, seed=(123 + i), return_task_id=True)
    model = SimpleMLP(num_classes=benchmark.n_classes, hidden_layers=2, hidden_size=400, drop_rate=0)
    tb_logger = TensorboardLogger(tb_log_dir="./tensorboard_logs/" + momentum + "-" + str(i))
    interactive_logger = InteractiveLogger()
    eval_plugin = EvaluationPlugin(
        accuracy_metrics(minibatch=False, epoch=False, experience=True, stream=False),
        loss_metrics(minibatch=False, epoch=False, experience=True, stream=False),
        forgetting_metrics(experience=True, stream=False),
        loggers=[interactive_logger, tb_logger]
    )  
    strategy = MyReplay(
        model=model,
        optimizer = SGD(model.parameters(), lr=0.01, momentum=float(momentum)),
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

