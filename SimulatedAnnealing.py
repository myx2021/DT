import abc
import copy
import json
import math
from random import random
import numpy as np
import time


from matplotlib import pyplot as plt


class SA:
    def __init__(self, initial_solution, T=100, T_min=0.05, alpha=0.99, max_time=12000, iters=1, iters_2=10, iters_3=10):
        self.T = T  # initial temperature
        self.T_min = T_min  # minimum temperature
        self.alpha = alpha  # the coefficient of temperature decreasing
        self.max_time = max_time  # time
        self.iter = iters  # the number of iterations in each temperature, which changes the radii of the orbits
        self.iter_2 = iters_2  # the number of iterations which changes the radii of the star
        self.iter_3 = iters_3
        self.history = {'T': [], 'solutions': [], 'scores': []}

        self.curr_solution = initial_solution
        self.new_solution = None  # placeholder
        self.score = self.evaluation()
        self.best_score = float('inf')  # choose the largest number as the initial best score

        self.time = 30 * 60

    @abc.abstractmethod
    def move(self):
        """Create a similar solution"""
        pass

    @abc.abstractmethod
    def move_2(self):
        """Create a similar solution"""
        pass

    @abc.abstractmethod
    def move_3(self):
        """Create a similar solution"""
        pass

    @abc.abstractmethod
    def evaluation(self):
        """Evaluate the solution, and give a score"""
        pass

    def check(self, new_score):  # Metropolis
        if new_score <= self.score:
            return True
        else:
            p = math.exp((new_score - self.score) / self.T)
            # print(p)
            if random() > p:
                return True
            else:
                return False

    def annealing(self):
        count = 0
        flag = 0
        flag_2 = 0

        bests = list()
        start = time.time()

        print(
            '================================================Simulation================================================\n')
        print("{:<15}{:<24}{:<24}{:<24}{:<24}".format('Iteration', 'Temperature', 'Best Score', 'Global Best Score',
                                                      'Num of Accepted Solution'))
        # print('Iteration\t\tTemperature\t\t\t\tBest Score\t\t\tGlobal Best Score\t\t\tNum of Accepted Solution')
        flag_c = 0  # convergence
        max_flag_c = 100
        while self.T > self.T_min and self.time > time.time() - start and count < 30000:

            solutions = [self.curr_solution]
            scores = [self.score]
            movement1 = 0.01
            movement2 = 0.1
            movement3 = 1
            temp_score = self.best_score
            for i in range(self.iter):  # iter is the number of iterations applied in each temperature
                self.move(movement1)
                for j in range(self.iter_2):
                    if self.move_2(movement2) == -1:
                        continue
                    for p in range(self.iter_3):
                        self.move_3(movement3)
                        new_score = self.evaluation()
                        if self.check(new_score):  # check whether accept the new solution or not
                            solution_copy = copy.deepcopy(self.curr_solution)
                            solutions.append(solution_copy)
                            scores.append(new_score)
                        if new_score < self.best_score:
                            temp_score = new_score
            if self.best_score == temp_score:
                flag_c = flag_c+1
            else:
                self.best_score = temp_score
                flag_c = 0

            if flag_c > max_flag_c:
                break

            bests.append(self.best_score)

            score_min = min(scores)  # find the smallest score
            scores = np.array(scores)
            indexes = np.argsort(scores)
            index = indexes[0]
            self.history['T'].append(self.T)
            copied_solution = copy.deepcopy(self.curr_solution)

            copied_solution['star1']['orbital_radius'] = copied_solution['star1']['orbital_radius'] * copied_solution[
                'orbit_coefficient_1_2']
            copied_solution['star2']['orbital_radius'] = copied_solution['star2']['orbital_radius'] * copied_solution[
                'orbit_coefficient_1_2']

            copied_solution['star1']['mass'] = copied_solution['star1']['mass'] * copied_solution[
                'mass_coefficient_1_2']
            copied_solution['star2']['mass'] = copied_solution['star2']['mass'] * copied_solution[
                'mass_coefficient_1_2']

            self.history['solutions'].append(copied_solution)
            self.history['scores'].append(score_min)

            # best solution in this temperature
            self.curr_solution = copy.deepcopy(solutions[index])

            # cooling
            self.T = self.T * self.alpha
            count += 1

            print("{:<15}{:<24}{:<24}{:<24}{:<24}".format(count, self.T, score_min, min(self.history['scores']),
                                                          (len(solutions) - 1) / self.iter / self.iter_2 / self.iter_3))
            # print(len(solutions))

            if count % 300 == 0:
                self.evaluation(True)
                # print(self.curr_solution)
                # print(copied_solution)
                f = open('Solutions.json', 'w')
                json_file = json.dumps(self.history, sort_keys=False, indent=4, separators=(',', ': '))
                f.write(json_file)
                f.write('\n')
                f.flush()
                f.close()

            # no accepted solutions in past 5 iterations
            accept_rate = (len(solutions) - 1) / (self.iter * self.iter_2 * self.iter_3)
            if accept_rate < 0.01:
                flag += 1
            else:
                flag = 0

            if flag > 10:
                break

            if accept_rate > 0.5:
                flag_2 += 1
            else:
                flag_2 = 0

            if flag_2 > 10:
                movement1 = movement1 / 2
                movement2 = movement2 / 2
                movement3 = movement3 / 2

                flag_2 = 0
        # best solution
        scores = self.history['scores']
        score_min = min(scores)  # find the smallest score
        index = scores.index(score_min)
        solution = self.history['solutions'][index]
        print(solution)
        if flag_c > max_flag_c:
            masses1 = list()
            masses2 = list()
            radii1 = list()
            radii2 = list()
            orb_radii1 = list()
            orb_radii2 = list()
            temperatures1 = list()
            temperatures2 = list()
            luminosities1 = list()
            luminosities2 = list()
            for i in range(1, max_flag_c-1):
                mass1 = np.array(self.history['solutions'][-i]['star1']['mass'])
                masses1.append(mass1)

                mass2 = np.array(self.history['solutions'][-i]['star2']['mass'])
                masses2.append(mass2)

                radius1 = np.array(self.history['solutions'][-i]['star1']['radius'])
                radii1.append(radius1)

                radius2 = np.array(self.history['solutions'][-i]['star2']['radius'])
                radii2.append(radius2)

                orb_radius1 = np.array(self.history['solutions'][-i]['star1']['orbital_radius'])
                orb_radii1.append(orb_radius1)

                orb_radius2 = np.array(self.history['solutions'][-i]['star2']['orbital_radius'])
                orb_radii2.append(orb_radius2)

                temperature1 = np.array(self.history['solutions'][-i]['star1']['temperature'])
                temperatures1.append(temperature1)

                temperature2 = np.array(self.history['solutions'][-i]['star2']['temperature'])
                temperatures2.append(temperature2)

                luminosity1 = np.array(self.history['solutions'][-i]['star1']['luminosity'])
                luminosities1.append(luminosity1)

                luminosity2 = np.array(self.history['solutions'][-i]['star2']['luminosity'])
                luminosities2.append(luminosity2)

            max1 = np.max(np.array(masses1)) - solution['star1']['mass'], solution['star1']['mass'] - np.min(np.array(masses1))
            max2 = np.max(np.array(masses2)) - solution['star2']['mass'], solution['star2']['mass'] - np.min(np.array(masses2))
            max3 = np.max(np.array(radii1)) - solution['star1']['radius'], solution['star1']['radius'] - np.min(np.array(radii1))
            max4 = np.max(np.array(radii2)) - solution['star2']['radius'], solution['star2']['radius'] - np.min(np.array(radii2))
            max5 = np.max(np.array(orb_radii1)) - solution['star1']['orbital_radius'], solution['star1']['orbital_radius'] - np.min(np.array(orb_radii1))
            max6 = np.max(np.array(orb_radii2)) - solution['star2']['orbital_radius'], solution['star2']['orbital_radius'] - np.min(np.array(orb_radii2))
            max7 = np.max(np.array(temperatures1)) - solution['star1']['temperature'], solution['star1']['temperature'] - np.min(np.array(temperatures1))
            max8 = np.max(np.array(temperatures2)) - solution['star2']['temperature'], solution['star2']['temperature'] - np.min(np.array(temperatures2))
            max9 = np.max(np.array(luminosities1)) - solution['star1']['luminosity'], solution['star1']['luminosity'] - np.min(np.array(luminosities1))
            max10 = np.max(np.array(luminosities2)) - solution['star2']['luminosity'], solution['star2']['luminosity']- np.min(np.array(luminosities2))

            # (0.0018995985822873873, 0.0039048129299021905)(0.0002517162539118212, 0.0005174276776717668)
            # (0.04338698495735738, 0.05314646443909699)(0.2748217448053736, 0.22038367903134137)
            # (0.0016720164881031119, 0.0008103145636880749)(0.012618017712412666, 0.0061151092647664385)
            # (10.632141308024075, -10.632141308024075)(-0.7971023970940223, 10.165417302824608)
            # (0.4574050512393768, 0.4.691916878266029)(0.29281410831785104,0.2009242169958244)

            # print(var1, var2, var3, var4, var5, var6, var7, var8, var9, var10)
            print(max1, max2, max3, max4, max5, max6, max7, max8, max9, max10)

        # plot
        plt.figure(figsize=(30, 8))
        # plt.gcf().set_size_inches(512 * 4 / 100, 512 * 2 / 100)
        plt.subplot(2, 3, 1)
        plt.margins(0, 0)
        plt.subplots_adjust(top=0.93, bottom=0.07, right=0.93, left=0.07)
        plt.subplots_adjust(hspace=0.3)
        plt.subplots_adjust(wspace=0.6)
        # plt.subplots_adjust(left=0.125, right=0.9, bottom=0.1, top=0.9)

        plt.ticklabel_format(axis="x", style="plain", scilimits=(0, 0))
        plt.xlabel('The number of iterations')
        plt.ylabel('Score')
        plt.plot(range(count), self.history['scores'])
        plt.title("Figure 1", fontsize='large', fontweight='bold')

        plt.subplot(2, 3, 2)
        plt.ticklabel_format(axis="x", style="plain", scilimits=(0, 0))
        plt.xlabel("The number of iterations")
        plt.ylabel("The value of radius")
        fig2_1 = plt.plot(range(count), [solution_index['star1']['radius'] for solution_index in self.history['solutions']],
                 label=r'Radius of primary star')
        fig2_2 = plt.plot(range(count), [solution_index['star2']['radius'] for solution_index in self.history['solutions']],
                 label=r'Radius of secondary star', color='red')
        plt.legend(loc='upper right')
        plt.title("Figure 2", fontsize='large', fontweight='bold')

        ax1 = plt.subplot(2, 3, 3)
        plt.ticklabel_format(axis="x", style="plain", scilimits=(0, 0))
        ax1.set_xlabel("The number of iterations")
        ax1.set_ylabel("Orbital radius of the primary star")

        plt.plot(range(count), [solution_index['star1']['orbital_radius'] for solution_index in self.history['solutions']],
                 label=r'Orbital radius (L: primary; R: secondary)')
        plt.legend(loc='upper right')
        ax2 = ax1.twinx()
        ax2.set_ylabel('Orbital radius of the secondary star')
        plt.plot(range(count), [solution_index['star2']['orbital_radius'] for solution_index in self.history['solutions']], label=r'Orbital radius (L: primary; R: secondary)')
        plt.title("Figure 3", fontsize='large', fontweight='bold')

        ax3 = plt.subplot(2, 3, 4)
        plt.ticklabel_format(axis="x", style="plain", scilimits=(0, 0))
        ax3.set_xlabel("The number of iterations")
        ax3.set_ylabel("Temperature of the primary star")
        plt.plot(range(count), [solution_index['star1']['temperature'] for solution_index in self.history['solutions']],
                 label=r'Temperature of primary star', color="red")
        plt.legend(loc='lower right')

        ax4 = ax3.twinx()
        plt.plot(range(count), [solution_index['star2']['temperature'] for solution_index in self.history['solutions']],
                 label=r'Temperature of secondary star')
        plt.legend(loc='upper right')
        ax4.set_ylabel("Temperature of the secondary star")
        plt.title("Figure 4", fontsize='large', fontweight='bold')

        ax5 = plt.subplot(2, 3, 5)
        plt.ticklabel_format(axis="x", style="plain", scilimits=(0, 0))
        ax5.set_xlabel("The number of iterations")
        ax5.set_ylabel("Luminosity of the primary star")
        plt.plot(range(count), [solution_index['star1']['luminosity'] for solution_index in self.history['solutions']],
                 label=r'Luminosity of primary star', color="red")
        plt.legend(loc='lower right')

        ax6 = plt.twinx()
        plt.plot(range(count), [solution_index['star2']['luminosity'] for solution_index in self.history['solutions']],
                 label=r'Luminosity of secondary star')
        plt.legend(loc='upper right')
        ax6.set_ylabel("Luminosity of the secondary star")
        plt.title("Figure 5", fontsize='large', fontweight='bold')

        ax7 = plt.subplot(2, 3, 6)
        plt.ticklabel_format(axis="x", style="plain", scilimits=(0, 0))
        ax7.set_xlabel("The number of iterations")
        ax7.set_ylabel("Mass of the primary star")

        plt.plot(range(count), [solution_index['star1']['mass'] for solution_index in self.history['solutions']],
                 label=r'Mass (L: primary; R: secondary)')
        plt.legend(loc='upper right')
        plt.title("Figure 6", fontsize='large', fontweight='bold')

        ax8 = plt.twinx()
        ax8.set_ylabel("Mass of the secondary star")
        plt.plot(range(count), [solution_index['star2']['mass'] for solution_index in self.history['solutions']])
        plt.show()
