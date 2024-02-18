import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

legends = ['Linear Reward Function', 'Transformable Gaussian Reward Function', '', '', '']

# add any folder directories here!
log_list = [
    pd.read_csv("trained_models/gaussian_model_no_pred/progress.csv"),
    pd.read_csv("trained_models/TGRF_3/progress.csv"),
]

logDicts = {}
for i in range(len(log_list)):
    logDicts[i] = log_list[i]

graphDicts = {0: 'eprewmean', 1: 'loss/value_loss'}

legendList = []

# Create line plots
for i in range(len(graphDicts)):
    plt.figure(i)
    plt.title(graphDicts[i])
    j = 0
    for key in logDicts:
        if graphDicts[i] not in logDicts[key]:
            continue
        else:
            x = logDicts[key]['misc/total_timesteps']
            y = logDicts[key][graphDicts[i]]

            # Select data points at every 5th index
            x_selected = x[::3]
            y_selected = y[::3]

            plt.plot(x_selected, y_selected)

            legendList.append(legends[j])
            print('avg', str(key), graphDicts[i], np.average(y))
        j = j + 1
    print('------------------------')

    plt.xlabel('total_timesteps')
    plt.ylabel(graphDicts[i])
    plt.title('Comparison between previous and our reward function')
    plt.legend(legendList, loc='upper left')
    legendList = []

plt.show()
