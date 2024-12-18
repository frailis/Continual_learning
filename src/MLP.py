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
        self.tb_writer = tb_writer 
        self.iter_count = 0  
        
    def _after_training_iteration(self, **kwargs):
        super()._after_training_iteration(**kwargs)
        self.iter_count = (self.iter_count + 1)

        results = self.eval(benchmark.test_stream[0])
        if 'Top1_Acc_Exp/eval_phase/test_stream/Task000/Exp000' in results:
            acc_task1 = results['Top1_Acc_Exp/eval_phase/test_stream/Task000/Exp000']
            self.tb_writer.add_scalar("Accuracy", acc_task1, self.iter_count)

    
    def _after_training(self, **kwargs):
        super()._after_training(**kwargs)

        results = self.eval(benchmark.test_stream)
        if 'Top1_Acc_Exp/eval_phase/test_stream/Task000/Exp000' in results:
            acc_task1 = results['Top1_Acc_Exp/eval_phase/test_stream/Task000/Exp000']
            self.tb_writer.add_scalar("END_Accuracy0", acc_task1)
        if 'Top1_Acc_Exp/eval_phase/test_stream/Task001/Exp001' in results:
            acc_task2 = results['Top1_Acc_Exp/eval_phase/test_stream/Task001/Exp001']
            self.tb_writer.add_scalar("END_Accuracy1", acc_task2)
        if 'Top1_Acc_Exp/eval_phase/test_stream/Task002/Exp002' in results:
            acc_task3 = results['Top1_Acc_Exp/eval_phase/test_stream/Task002/Exp002']
            self.tb_writer.add_scalar("END_Accuracy2", acc_task3)
        if 'Top1_Acc_Exp/eval_phase/test_stream/Task003/Exp003' in results:
            acc_task4 = results['Top1_Acc_Exp/eval_phase/test_stream/Task003/Exp003']
            self.tb_writer.add_scalar("END_Accuracy3", acc_task4)
        if 'Top1_Acc_Exp/eval_phase/test_stream/Task004/Exp004' in results:
            acc_task5 = results['Top1_Acc_Exp/eval_phase/test_stream/Task004/Exp004']
            self.tb_writer.add_scalar("END_Accuracy4", acc_task5)




n_seeds = int(sys.argv[1])
memory = int(sys.argv[2])
momentum_list=[0, 0.1, 0.5, 0.9]

for j, momentum in enumerate(sys.argv[3:]):
    for i in range(0, n_seeds):
        benchmark = SplitMNIST(n_experiences = 5, seed=(123 + i), return_task_id=True)
        model = SimpleMLP(num_classes=benchmark.n_classes, hidden_layers=2, hidden_size=400, drop_rate=0)
        tb_logger = TensorboardLogger(tb_log_dir="./tensorboard_logs/2layer-" + str(memory) + "-" + momentum + "-" + str(i))
        interactive_logger = InteractiveLogger()
        eval_plugin = EvaluationPlugin(
            accuracy_metrics(minibatch=True, epoch=True, experience=True, stream=True),
            loss_metrics(minibatch=True, epoch=True, experience=True, stream=True),
            forgetting_metrics(experience=True, stream=True),
            loggers=[interactive_logger, tb_logger]
        )  
        strategy = MyReplay(
            model=model,
            optimizer = SGD(model.parameters(), lr=0.01, momentum=float(momentum)),
            criterion = CrossEntropyLoss(),
            mem_size = memory,
            train_mb_size = 256,
            eval_mb_size = 256,
            train_epochs = 10,
            evaluator = eval_plugin,
            eval_every = 1,
            tb_writer = tb_logger.writer
        )   
        print("Experiment Start")

        #f = open("LOG.txt", "a")

        for experience in benchmark.train_stream: 
            #f.write(str(strategy.optimizer.state_dict().get('state')))
            #strategy.optimizer = SGD(model.parameters(), lr=0.01, momentum=float(momentum))
            #f.write(str(strategy.optimizer.state_dict().get('state')))
            res = strategy.train(experience)  

        #f.close()
