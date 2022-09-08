# -*- coding: utf-8 -*-
from __future__ import print_function
import math
import random
import numpy as np
import matplotlib.pyplot as plt
from matplotlib import animation
# from SA2 import Annealer
from matplotlib.animation import FuncAnimation
from scipy.interpolate import interp1d
from SimulatedAnnealing import SA
from RCMa_system import RCMA
from Calculation import draw_RVs, draw, draw_RVs_compare, date_to_phase, interpolation_RVs

L0 = 3.0128 * math.pow(10, 28)
L_sun = 3.828 * math.pow(10, 26)
R_sun = 695700000.0  # meter
M_sun = 1.98847 * math.pow(10, 30)
T_sun = 5778.0  # K
pc = 3.0857 * math.pow(10, 16)
au = 1.495978707 * math.pow(10, 11)
G = 6.67430 * math.pow(10, -11)


def read_photometric_data(filename):
    period = float()
    Z = float()
    data = list()

    with open(filename, 'r') as f:

        line = f.readline()

        while line:
            if '#' in line or line == '\n':
                parameters = line.split(' ')
                match parameters[0]:
                    case '#P':
                        period = float(parameters[1])
                    case '#Z':
                        Z = float(parameters[1].strip())

                line = f.readline()
                continue

            time_light = line.split(' ')[:2]
            data.append(time_light)

            line = f.readline()

    # print(data)
    photometric_data = [math.pow(10, -0.4 * float(p[1])) for p in data]
    date = [float(t[0]) for t in data]
    phases = date_to_phase(date, period, Z)

    phases = np.array(phases)
    phases_index = phases.argsort()
    phases = phases[phases_index]

    photometric_data = np.array(photometric_data)
    photometric_data = photometric_data[phases_index]
    photometric_data_inverse = photometric_data
    photometric_data = np.hstack((photometric_data, photometric_data_inverse))
    phases = np.hstack((phases, [p + max(phases) for p in phases]))

    draw(phases, photometric_data, 'Phase', "Flux")

    # phases, photometric_data, f = interpolation_photometric(phases, photometric_data)
    # draw(phases, photometric_data, 'Phase', "Photometric data")

    # print(photometric_data)
    # print(time)
    return period, Z, date, phases, photometric_data


def read_RV_data(filename):
    date_JD = list()
    phase1 = list()
    phase2 = list()
    rv1 = list()
    rv2 = list()
    with open(filename, 'r') as f:

        line = f.readline()
        # print(line)

        while line:
            if line == '\n':
                line = f.readline()
                continue

            parameters = line.split(' ')

            date_JD.append(float(parameters[3]))
            p = float(parameters[4]) - 0.031
            if p < 0:
                p += 1
            phase1.append(p)
            rv1.append(float(parameters[5]))

            if len(parameters) == 7:
                phase2.append(p)
                rv2.append(float(parameters[6]))

            line = f.readline()

    phase1 = np.array(phase1)
    phase2 = np.array(phase2)
    rv1 = np.array(rv1)
    rv2 = np.array(rv2)
    phase_index1 = phase1.argsort()
    phase_index2 = phase2.argsort()
    phase1 = phase1[phase_index1]
    phase2 = phase2[phase_index2]
    rv1 = rv1[phase_index1]
    rv2 = rv2[phase_index2]

    rv1 = np.hstack((rv1, np.hstack((rv1, rv1))))
    rv2 = np.hstack((rv2, np.hstack((rv2, rv2))))
    phase1 = np.hstack((phase1 - 1.0, np.hstack((phase1, phase1[0:len(phase1)] + 1.0))))
    phase2 = np.hstack((phase2 - 1.0, np.hstack((phase2, phase2[0:len(phase2)] + 1.0))))
    # draw_RVs(phase1, phase2, rv1, rv2, 'Phase [-1, 1]', "Radial Velocity (km/s)")

    # print(phase1)
    # print(phase2)

    rv1_max_list = []
    rv2_max_list = []
    # for i in np.linspace(3, 35, num=50):
    # print(i)
    rv1_max, fitting_phase1, fitting_rv1, f = interpolation_RVs(phase1, rv1)
    rv2_max, fitting_phase2, fitting_rv2, f = interpolation_RVs(phase2, rv2)
    rv1_max_list.append((np.max(fitting_rv1) + np.min(fitting_rv1)) / 2)
    rv2_max_list.append((np.max(fitting_rv2) + np.min(fitting_rv2)) / 2)
    # if i % 5 == 0:
    #     draw_RVs(phase1, phase2, rv1, rv2, fitting_phase1, fitting_phase2, fitting_rv1, fitting_rv2, 'Phase [-1, 2]',
    #          "Observed RVs (km/s)", 'Phase [0, 1]', "Fitted RVs (km/s)")
    # print(i)
    # print(rv1_max, rv2_max)
    # print((np.max(fitting_rv1) + np.min(fitting_rv1)) / 2)
    # print((np.max(fitting_rv2) + np.min(fitting_rv2)) / 2)

    # plt.figure(figsize=(30, 4))
    # plt.ticklabel_format(axis="x", style="plain", scilimits=(0, 0))
    # plt.xlabel("Highest Degree")
    # plt.ylabel("Difference between RV1_max and RV2_max")
    # plt.plot(np.linspace(3, 35, num=50), rv1_max_list, label=r"Median value of RV 1")
    # plt.plot(np.linspace(3, 35, num=50), rv2_max_list, label=r"Median value of RV 2")
    # plt.legend()
    # plt.show()

    draw_RVs(phase1, phase2, rv1, rv2, fitting_phase1, fitting_phase2, fitting_rv1, fitting_rv2, 'Phase [-1, 2]', "Observed RVs (km/s)", 'Phase [0, 1]', "Fitted RVs (km/s)")

    # print("Velocity of Star 1: ", rv1_max)
    # print("Velocity of Star 2: ", rv2_max)
    return date_JD, phase1, phase2, rv1, rv2, fitting_rv1, fitting_rv2, rv1_max, rv2_max, f


def main():
    random.seed(a=None, version=2)

    # read photometric data
    period, Z, time, phases_p, photometric_data = read_photometric_data('V.dat')

    # read RV data
    _, phase_rv1, phase_rv2, rv1, rv2, fitting_rv1, fitting_rv2, rv1_max, rv2_max, f = read_RV_data("RV.txt")

    # generate an initial solution from previous research
    # M, error range of M, R, error range of R, L, error range of L
    # since the T = 2*pi*r is same for these two orbits, the orb_r1/orb_r2 = rv1 / rv2
    ratio_r1_r2 = rv1_max / rv2_max  # use the maximum radial velocity as the initial velocity
    init_orb1 = rv1_max * 1000.0 * period * 24 * 60 * 60 / 2 / math.pi / R_sun
    init_orb2 = rv2_max * 1000.0 * period * 24 * 60 * 60 / 2 / math.pi / R_sun
    orbit_coefficient = init_orb2  # the initial value
    orb_r1 = ratio_r1_r2
    orb_r2 = 1
    orb_r3 = None  # placeholder
    orbital_radius_error = None  # placeholder

    # since the m1*v1**2/r1 = m2*v2**2/r2, the m1/m2 = rv2 / rv1
    ratio_m1_m2 = rv2_max / rv1_max
    init_m1 = math.pow(rv2_max * 1000.0, 2) * math.pow((init_orb1 * R_sun + init_orb2 * R_sun), 2) / (init_orb2 * R_sun * G) / M_sun
    init_m2 = math.pow(rv1_max * 1000.0, 2) * math.pow((init_orb1 * R_sun + init_orb2 * R_sun), 2) / (init_orb1 * R_sun * G) / M_sun
    mass_coefficient = init_m2
    m1 = ratio_m1_m2
    m2 = 1
    m3 = None  # placeholder

    # radius, unit: radius of the Sun
    r1 = 1.78
    r2 = 1.22
    r3 = 0.83

    # temperature
    T1 = 7300  # K
    T2 = 4350
    T3 = None  # placeholder
    L1 = math.pow(r1, 2) * math.pow(T1 / T_sun, 4)
    L2 = math.pow(r2, 2) * math.pow(T2 / T_sun, 4)
    L3 = None

    print('============================================================Initial parameters================================================\n')
    # print('Property\tRadius\tMass\tOrbital radius\tTemperature\tLuminosity')

    print("{:<15}{:<15}{:<24}{:<24}{:<24}{:<24}{:<24}".format('Property', 'Radius', 'Mass', 'Orbital radius', 'Linear velocity', 'Temperature', 'Luminosity'))
    print("{:<15}{:<15}{:<24}{:<24}{:<24}{:<24}{:<24}".format('Star1', r1, m1 * mass_coefficient, orb_r1 * orbit_coefficient, rv1_max, T1, L1))
    print("{:<15}{:<15}{:<24}{:<24}{:<24}{:<24}{:<24}".format('Star2', r2, m2 * mass_coefficient, orb_r2 * orbit_coefficient, rv2_max, T2, L2))
    print("{:<15}{:<15}{:<24}{:<24}{:<24}{:<24}{:<24}\n".format('Star3', r3, "None", "None", "None", "None", "None"))

    star1 = {'mass': m1, 'radius': r1, 'temperature': T1, 'luminosity': L1,
             'orbital_radius': orb_r1, 'mass_error': None, 'radius_error': None,'temperature_error': None, 'orbital_radius_error': orbital_radius_error}
    star2 = {'mass': m2, 'radius': r2, 'temperature': T2, 'luminosity': L2,
             'orbital_radius': orb_r2, 'mass_error': None, 'radius_error': None,'temperature_error': None, 'orbital_radius_error': orbital_radius_error}
    star3 = {'mass': m3, 'radius': r3, 'temperature': T3, 'luminosity': 0.4,
             'orbital_radius': orb_r3, 'mass_error': None, 'radius_error': None,'temperature_error': None, 'orbital_radius_error': orbital_radius_error}

    init_solution = {'star1': star1, 'star2': star2, 'star3': star3, 'period': period, 'distance': 44 * pc / R_sun,
                     'mass_coefficient_1_2': mass_coefficient, 'orbit_coefficient_1_2': orbit_coefficient}

    # generate an object of the simulated annealing class
    sa = RCMA(init_solution, phases_p, photometric_data, phase_rv1, phase_rv2, rv1, rv2)
    sa.annealing()
    # sa.set_schedule(sa.auto(1))


if __name__ == '__main__':
    # print(date_to_JD(2010, 8, 9, 17.0, 16.0, 00.4))
    # print(JD_to_date(2455418.21421))
    # print(JD_to_date(2455418.22554))
    # print(JD_day_to_hms(.21421))

    main()
