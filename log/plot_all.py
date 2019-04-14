import matplotlib.pyplot as plt
import numpy as np

import os
import os.path as osp
import glob

def plot_one(path):
    run_name = path.split('/')[1]
    title = input("Name {}: ".format(run_name))

    if title == '_s':
        return None, None, None

    else:
        arr = np.array(np.genfromtxt(path, delimiter=',')).T

        x = arr[0]
        y = arr[1]
        return x,y,title

def main():
    for path in glob.glob('./*/epoch/*/Val_Loss.csv'):
        x_range, y_range, leg = plot_one(path)
        if x_range is None:
            continue

        else:
            plt.plot(x_range, y_range, label=leg)

    plt.legend()
    plt.xlabel("Train Steps")
    plt.ylabel("Avg. Log Perplexity")
    plt.savefig("Val Loss Plot")

if __name__ == '__main__':
    main()