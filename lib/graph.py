"""
Graph function
"""

import matplotlib.pyplot as plt
import numpy as np
from paths import *

"""
Show bar chart
"""
def barChart(x,y,label=False,name='basic-bar'):
    y_pos = np.arange(len(x))

    plt.bar(y_pos, y)
    if label:
        plt.xticks(y_pos, x)
    plt.yticks()
    plt.savefig(graph_path+name+'.png')
    plt.close()


"""
Show line chart
"""


def lineChart(x, y=None, x_label = None,y_label = None, title='Line Chart'):

    if y is None:
        plt.plot(x)
    else:
        plt.plot(x, y)

    if x_label:
        plt.xlabel(x_label)

    if y_label:
        plt.ylabel(y_label)

    plt.title(title)
    plt.savefig(graph_path+title+'.png')
    plt.close()
