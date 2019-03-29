# -*- coding:UTF-8 -*-
import os
from matplotlib import pyplot as plt
import json

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

def save_word_dict(word_index):
    # save
    filename = 'dict_tweets'
    with open(filename + '.json', 'w') as outfile:
        # json.dump(word_index,outfile,ensure_ascii=False)
        json.dump(word_index, outfile)
        # outfile.write('\n')

def texts_stat(texts, sequences, word_index, dict_name='dict_tweets.json', verbose=1):
    # raw texts
    # after tokenizer
    # map words to tokens
    sum_of_char = sum([len(t) for t in texts])
    if verbose:
        print('Found %s texts.' % len(texts))
        print('Averg chars of a tweet %s' % (sum_of_char // len(texts)))

        print('Averg words of a tweet %s' % ( sum([len(t) for t in sequences])// len(sequences)))

    if not os.path.exists('./'+dict_name):
        save_word_dict(word_index)

# load dictionary
def load_dict(path):
    with open(path, 'r') as f:
        dict = json.load(f)
    print('Open Word Dict, Length: %s' % len(dict))
    return dict

def read_file(path, head='Subject:'):
    args = {} if sys.version_info < (3,) else {'encoding': 'latin-1'}
    with open(path, **args) as f:
        t = f.read()
        i = t.find(head)  # skip header
        if 0 < i:
            t = t[i:]
    return t



