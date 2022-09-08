import abc
import copy
import json
import math
from random import random
import numpy as np
import time


from matplotlib import pyplot as plt


class SA:
    def __init__(self, initial_solution, T=100, T_min=0.05, alpha=0.99, max_time=600, iters=10, iters_2=5, iters_3=5):
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

        start = time.time()

        print(
            '================================================Simulation================================================\n')
        print("{:<15}{:<24}{:<24}{:<24}{:<24}".format('Iteration', 'Temperature', 'Best Score', 'Global Best Score',
                                                      'Num of Accepted Solution'))
        # print('Iteration\t\tTemperature\t\t\t\tBest Score\t\t\tGlobal Best Score\t\t\tNum of Accepted Solution')
        while self.T > self.T_min and self.time > time.time() - start and count < 300:

            solutions = [self.curr_solution]
            scores = [self.score]
            for i in range(self.iter):  # iter is the number of iterations applied in each temperature
                movement1 = 0.01
                movement2 = 0.1
                movement3 = 0.1
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
                                                          (len(solutions) - 1)/ self.iter / self.iter_2 / self.iter_3))

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
            if len(solutions) < 5:
                flag += 1
            else:
                flag = 0

            if flag > 5:
                break

        # best solution
        scores = self.history['scores']
        score_min = min(scores)  # find the smallest score
        index = scores.index(score_min)
        solution = self.history['solutions'][index]

        print(solution)

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

