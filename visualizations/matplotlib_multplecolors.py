# -*- coding: utf-8 -*-
"""
Created on Tue Jun 19 23:15:54 2018

@author: David
"""


#Define a helper function (this a bare-bones one, more bells and whistles can be added). 
#This code is a slight refactoring of this example from the documentation.

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.collections import LineCollection
from matplotlib.colors import ListedColormap, BoundaryNorm

def threshold_plot(ax, x, y, threshv1,threshv2, color, overcolor,finalcolor):
    """
    Helper function to plot points above a threshold in a different color

    Parameters
    ----------
    ax : Axes
        Axes to plot to
    x, y : array
        The x and y values

    threshv : float
        Plot using overcolor above this value

    color : color
        The color to use for the lower values

    overcolor: color
        The color to use for values over threshv

    """
    # Create a colormap for red, green and blue and a norm to color
    # f' < -0.5 red, f' > 0.5 blue, and the rest green
    cmap = ListedColormap([color, overcolor,finalcolor])
    norm = BoundaryNorm([np.min(y), threshv1,threshv2, np.max(y)], cmap.N)

    # Create a set of line segments so that we can color them individually
    # This creates the points as a N x 1 x 2 array so that we can stack points
    # together easily to get the segments. The segments array for line collection
    # needs to be numlines x points per line x 2 (x and y)
    points = np.array([x, y]).T.reshape(-1, 1, 2)
    segments = np.concatenate([points[:-1], points[1:]], axis=1)

    # Create the line collection object, setting the colormapping parameters.
    # Have to set the actual values used for colormapping separately.
    lc = LineCollection(segments, cmap=cmap, norm=norm)
    lc.set_array(y)

    ax.add_collection(lc)
    ax.set_xlim(np.min(x), np.max(x))
    ax.set_ylim(np.min(y)*1.1, np.max(y)*1.1)
    return lc

#USage example.
fig, ax = plt.subplots()

x = np.linspace(0, 3 * np.pi, 500)
y = np.sin(x)

lc = threshold_plot(ax, x, y, .5,.75, 'k', 'r','g')
ax.axhline(.75, color='k', ls='--')
ax.axhline(.5, color='b', ls='--')
lc.set_linewidth(3)


## opc 2
import matplotlib.pyplot as plt
import numpy as np

fig, ax = plt.subplots()

ys = np.random.rand(10)

threshold = 0.5

ax.axhline(y=threshold, color='r', linestyle=':')
ax.plot(ys)

greater_than_threshold = [i for i, val in enumerate(ys) if val>threshold]
ax.plot(greater_than_threshold, ys[greater_than_threshold], 
        linestyle='none', color='r', marker='o')

plt.show()