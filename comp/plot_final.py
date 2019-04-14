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
    final_x = []
    final_y = []
    for path in glob.glob('./*/epoch/*/Train_Loss.csv'):
        x_range, y_range, leg = plot_one(path)
        if x_range is None:
            continue

        else:
            try:
                final_y.append(y_range[24])
            except:
                final_y.append(y_range[23])
            final_x.append(float(leg))

    plt.scatter(np.log(1-np.array(final_x)), final_y)

    plt.xlabel("Log Connections Remaining")
    plt.ylabel("Avg. Log Perplexity")
    plt.title("Train Loss after 48k training steps")
    plt.savefig("Final Train Loss Plot")

if __name__ == '__main__':
    main()