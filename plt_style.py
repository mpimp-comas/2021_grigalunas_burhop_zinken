#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
My favourite Matplotlib styles."""

import matplotlib.pyplot as plt

plt.style.use("seaborn-white")
# plt.style.use("seaborn-pastel")
# plt.style.use("seaborn-whitegrid")
plt.style.use("seaborn-poster")  # seaborn-talk
plt.style.use("seaborn-ticks")
plt.rcParams['axes.titlesize'] = 25
plt.rcParams['axes.labelsize'] = 25
plt.rcParams['xtick.labelsize'] = 20
plt.rcParams['ytick.labelsize'] = 20
plt.rcParams['legend.fontsize'] = 20
