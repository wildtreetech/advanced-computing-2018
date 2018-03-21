import csv

import numpy as np
import matplotlib.pyplot as plt

import subprocess
from tempfile import mkstemp

from sklearn.tree import export_graphviz

from IPython.core.display import HTML


def plot_surface(clf, X, y, n_steps=250, subplot=None, show=True,
                 ylim=None, xlim=None):
    if subplot is None:
        fig = plt.figure()
    else:
        plt.subplot(*subplot)

    if xlim is None:
        xlim = X[:, 0].min(), X[:, 0].max()
    if ylim is None:
        ylim = X[:, 1].min(), X[:, 1].max()
    xx, yy = np.meshgrid(np.linspace(xlim[0], xlim[1], n_steps),
                         np.linspace(ylim[0], ylim[1], n_steps))

    if hasattr(clf, "decision_function"):
        z = clf.decision_function(np.c_[xx.ravel(), yy.ravel()])
    else:
        z = clf.predict_proba(np.c_[xx.ravel(), yy.ravel()])[:, 1]

    z = z.reshape(xx.shape)
    plt.contourf(xx, yy, z, alpha=0.8, cmap=plt.cm.RdBu_r)
    plt.scatter(X[:, 0], X[:, 1], c=y)
    plt.xlim(*xlim)
    plt.ylim(*ylim)

    if show:
        plt.show()


def draw_tree(clf, feature_names, svg_name=None, **kwargs):
    _, name = mkstemp(suffix='.dot')
    if svg_name is None:
        _, svg_name = mkstemp(suffix='.svg')
    export_graphviz(clf, out_file=name,
                    feature_names=feature_names,
                    **kwargs)
    command = ["dot", "-Tsvg", name, "-o", svg_name]
    subprocess.check_call(command)
    return HTML(open(svg_name).read())


def load_wine(data_dir='../../data'):
    X = []
    y = []

    red = csv.reader(open(data_dir + '/winequality-red.csv'),
                     delimiter=';', quotechar='"')
    # skip header
    next(red)
    for l in red:
        y.append(float(l[-1]))
        X.append(list(map(float, l[:-1])))
    white = csv.reader(open(data_dir + '/winequality-white.csv'),
                       delimiter=';', quotechar='"')
    # skip header
    next(white)
    for l in white:
        y.append(float(l[-1]))
        X.append(list(map(float, l[:-1])))
    X = np.array(X)
    y = np.array(y)
    return X, y


def plot_loss(est, X_test, y_test, ax=None, label='',
              train_color=None, test_color=None, alpha=1.0, ylim=(0, 0.7)):
    n_estimators = len(est.estimators_)
    test_dev = np.empty(n_estimators)

    for i, pred in enumerate(est.staged_predict(X_test)):
        test_dev[i] = est.loss_(y_test, pred)

    if ax is None:
        fig = plt.figure()
        ax = plt.gca()

    ax.plot(np.arange(n_estimators) + 1, test_dev, color=test_color,
            label='Test %s' % label, linewidth=2, alpha=alpha)
    ax.plot(np.arange(n_estimators) + 1, est.train_score_, color=train_color,
            label='Train %s' % label, linewidth=2, alpha=alpha)
    ax.set_ylabel('Loss')
    ax.set_xlabel('n_estimators')
    ax.set_ylim(ylim)
    return test_dev, ax
