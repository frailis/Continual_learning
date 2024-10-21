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
    end_values = [[], [], [], [], []]

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

        end_values[0].append(event_acc.Scalars("END_Accuracy0")[-1].value)
        end_values[1].append(event_acc.Scalars("END_Accuracy1")[-1].value)
        end_values[2].append(event_acc.Scalars("END_Accuracy2")[-1].value)
        end_values[3].append(event_acc.Scalars("END_Accuracy3")[-1].value)
        end_values[4].append(event_acc.Scalars("END_Accuracy4")[-1].value)

    for value_list in end_values:
        print(str(np.average(value_list)) + " - " + str(np.max(value_list) - np.average(value_list)))
    

    
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
    string = './tensorboard_logs/' + sys.argv[i]
    logs.append(string)
tag = 'Accuracy'  
plot_logs(logs, tag, xlabel='Iterations', ylabel='Accuracy', title='Accuracy for the first task M=0.0')