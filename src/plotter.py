import os
import numpy as np
import matplotlib.pyplot as plt
from tensorboard.backend.event_processing.event_accumulator import EventAccumulator
import sys

def plot_logs(log_dirs, tag, xlabel='Iterations', ylabel='Value', title='Average'):
    scalar_values = []
    common_steps = None
    sum_values = None
    i=0

    for log_dir in log_dirs:
        i += 1
        event_acc = EventAccumulator(log_dir)
        event_acc.Reload()

        events = event_acc.Scalars(tag)
        steps = np.array([e.step for e in events])
        values = np.array([e.value for e in events])

        if common_steps is None:
            common_steps = steps
        else:
            common_steps = np.union1d(common_steps, steps)
        
        aligned_values = np.interp(common_steps, steps, values)
        scalar_values.append(aligned_values)
        if sum_values is None:
            sum_values = aligned_values
        else:
            sum_values = sum_values + aligned_values

    
    min_values = np.min(scalar_values, axis=0)
    max_values = np.max(scalar_values, axis=0)
    mean_values = sum_values/i

    # Plot the values
    plt.figure(figsize=(10, 6))
    plt.plot(common_steps, mean_values, label='Mean', color='blue', linestyle='-')
    plt.fill_between(common_steps, min_values, max_values, facecolor='powderblue', interpolate=True)

    # Customize the plot
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    plt.title(title)
    plt.legend()
    plt.grid(True)
    plt.show()

logs = []
for i in range(1, len(sys.argv)):
    str = './tensorboard_logs/' + sys.argv[i]
    logs.append(str)
tag = 'Accuracy'  
plot_logs(logs, tag, xlabel='Iterations', ylabel='Accuracy', title='Accuracy for the first task M=0.1')