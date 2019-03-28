# -*- coding:UTF-8 -*-
import os
from matplotlib import pyplot as plt

def plot_line(*args, **kw):
    """

    :param args: x,y,...
    :param kw: name, xlabel, ylabel,...
    :return:
    """
    name = 'line_chart'
    if 'name' in kw:
        name = kw['name']
        kw.pop('name')
    xlabel = ''
    if 'xlabel' in kw:
        xlabel = kw['xlabel']
        kw.pop('xlabel')
    ylabel = ''
    if 'ylabel' in kw:
        ylabel = kw['ylabel']
        kw.pop('ylabel')

    if 'c' and 'color' not in kw:
        kw['color'] = 'r'

    fig = plt.figure()
    plt.title(name)
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    plt.plot(*args,**kw)

    plt.grid()
    plt.show()

#import numpy as np
#a = np.arange(36)
#plot_line(a,name='car')

def create_path(path=''):
    if os.path.exists(path) is not True:
        os.mkdir(path)
        print('Create the path: '+path)