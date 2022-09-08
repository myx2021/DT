import math

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.animation import FuncAnimation
from scipy.interpolate import interp1d


def date_to_phase(date, period, Z):
    phases = list()
    for single_day in date:
        phase = (single_day - Z) / period
        phase = phase - math.floor(phase)  # only keep integer
        phases.append(phase)
    # print(phases)
    return phases


def draw(x, y, x_label, y_label):
    plt.figure(figsize=(30, 6))
    plt.ticklabel_format(axis="x", style="plain", scilimits=(0, 0))
    plt.xlabel(x_label, fontsize=18, labelpad=10)
    plt.ylabel(y_label, fontsize=18, labelpad=30)
    plt.xticks(fontsize=15)
    plt.yticks(fontsize=15)
    plt.plot(x, y)
    plt.show()


def interpolation_RVs(x, y):

    f = np.polyfit(x, y, 14)
    # f = np.polyfit(x, y, degree)
    p1 = np.poly1d(f)

    x_pred = np.linspace(0, 1, num=10000)
    y_pred = p1(x_pred)

    y_max = np.max(y_pred)
    y_min = np.min(y_pred)

    y_median = (y_max + y_min) / 2
    y_real_max = y_max - y_median

    return y_real_max, x_pred, y_pred, f


def interpolation_photometric(x, y):
    f = interp1d(x, y, kind='cubic', fill_value="extrapolate")
    x_pred = np.linspace(0, 1, num=1000)
    y_pred = f(x_pred)

    return x_pred, y_pred, f


def draw_compare(x, y1, y2, x_label, y_label):
    plt.figure(figsize=(30, 4))
    plt.ticklabel_format(axis="x", style="plain", scilimits=(0, 0))
    plt.xlabel(x_label)
    plt.ylabel(y_label)
    plt.plot(x, y1)
    plt.plot(x, y2)
    plt.show()


def draw_RVs(phase1, phase2, rv1, rv2, fitting_phase1, fitting_phase2, fitting_rv1, fitting_rv2, x_label, y_label, fitting_x, fitting_y):
    plt.figure(figsize=(30, 8))
    # plt.title("RVs VS. Phases")
    plt.subplot(2, 1, 1)
    plt.subplots_adjust(hspace=0.3)
    plt.ticklabel_format(axis="x", style="plain", scilimits=(0, 0))
    plt.xlabel(x_label, fontsize=18)
    plt.ylabel(y_label, fontsize=18)
    plt.xticks(fontsize=15)
    plt.yticks(fontsize=15)

    plt.scatter(phase1, rv1, label=r'Observed RV1')
    plt.scatter(phase2, rv2, label=r'Observed RV2')

    plt.plot(phase1, rv1, label=r'Observed RV1 line')
    plt.plot(phase2, rv2, label=r'Observed RV2 line')

    plt.legend(loc='upper right')

    plt.subplot(2, 1, 2)
    plt.ticklabel_format(axis="x", style="plain", scilimits=(0, 0))

    plt.xlabel(fitting_x, fontsize=18)
    plt.ylabel(fitting_y, fontsize=18)
    plt.xticks(fontsize=15)
    plt.yticks(fontsize=15)

    plt.plot(fitting_phase1, fitting_rv1, label=r'Fitting RV1 line')
    plt.plot(fitting_phase2, fitting_rv2, label=r'Fitting RV2 line')

    plt.legend(loc='upper right')

    plt.show()


def draw_RVs_p_compare(phase1, phase2, phases_p, rv1, rv2, rv1_simulated, rv2_simulated, x_label, y_label, x_p, y1_p,
                       y2_p, x_label_p, y_label_p):
    plt.figure(figsize=(30, 8))
    plt.title("RVs and photometric data VS. Phases")
    plt.subplot(2, 1, 1)
    plt.subplots_adjust(hspace=0.3)
    plt.subplots_adjust(wspace=0.3)

    plt.ticklabel_format(axis="x", style="plain", scilimits=(0, 0))

    plt.xlabel(x_label_p, fontsize=18)
    plt.ylabel(y_label_p, fontsize=18)
    plt.xticks(fontsize=15)
    plt.yticks(fontsize=15)

    plt.plot(x_p, y2_p, label=r'Observed LC')
    plt.plot(x_p, y1_p, label=r'Predicted LC')
    plt.legend(loc='upper right')

    plt.subplot(2, 1, 2)
    plt.ticklabel_format(axis="x", style="plain", scilimits=(0, 0))
    plt.xlabel(x_label, fontsize=18)
    plt.ylabel(y_label, fontsize=18)
    plt.xticks(fontsize=15)
    plt.yticks(fontsize=15)
    rv1 = rv1[len(rv1) // 3:]
    rv2 = rv2[len(rv2) // 3:]
    phase1 = phase1[len(phase1) // 3:]
    phase2 = phase2[len(phase2) // 3:]
    plt.scatter(phase1, rv1, label=r'Observed RV1')
    plt.scatter(phase2, rv2, label=r'Observed RV2')
    plt.plot(phase1, rv1, label=r'Observed RV1 line')
    plt.plot(phase2, rv2, label=r'Observed RV2 line')
    plt.plot(phases_p, rv1_simulated, label=r'Simulated RV1')
    plt.plot(phases_p, rv2_simulated, label=r'Simulated RV2')
    plt.legend(loc='upper right')
    plt.show()


def draw_RVs_compare(phase1, phase2, phases_p, rv1, rv2, rv1_simulated, rv2_simulated, x_label, y_label):
    plt.figure(figsize=(30, 4))
    plt.ticklabel_format(axis="x", style="plain", scilimits=(0, 0))
    plt.xlabel(x_label)
    plt.ylabel(y_label)
    rv1 = rv1[len(rv1) // 3:]
    rv2 = rv2[len(rv2) // 3:]
    phase1 = phase1[len(phase1) // 3:]
    phase2 = phase2[len(phase2) // 3:]
    plt.scatter(phase1, rv1, label=r'Observed RV1')
    plt.scatter(phase2, rv2, label=r'Observed RV2')
    plt.plot(phase1, rv1, label=r'Observed RV1 line')
    plt.plot(phase2, rv2, label=r'Observed RV2 line')
    plt.plot(phases_p, rv1_simulated, label=r'Simulated RV1')
    plt.plot(phases_p, rv2_simulated, label=r'Simulated RV2')
    plt.legend(loc='upper right')
    plt.show()


def drawAnimation(positions_1, positions_2):
    plt.rcParams['font.sans-serif'] = ['KaiTi']
    plt.rcParams['axes.unicode_minus'] = False

    fig, ax = plt.subplots(figsize=(6, 6))
    fig.set_tight_layout(True)

    ax.set_xlim(-20, 20)
    ax.set_ylim(-20, 20)
    ax.set_xlabel('X  Unit: Radius of the sun')
    ax.set_ylabel('Y  Unit: Radius of the sun')

    def update(i):
        x = list()
        y = list()
        if i * 10 < len(positions_1):
            for j in positions_1[0:i * 50]:
                x.append(j[0])
                y.append(j[1])
        ax.scatter(x, y, s=50)

        x2 = list()
        y2 = list()
        if i * 10 < len(positions_2):
            for j in positions_2[0:i * 50]:
                x2.append(j[0])
                y2.append(j[1])

        ax.scatter(x2, y2, s=80)

        return ax

    anim = FuncAnimation(fig, update, frames=np.arange(0, 20), interval=100, repeat=False)

    plt.show()
    # anim.save("test.gif", writer='pillow')
    return 0
