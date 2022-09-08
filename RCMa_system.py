import math
import random
import numpy as np
from Calculation import draw, drawAnimation, draw_compare, draw_RVs, draw_RVs_compare, draw_RVs_p_compare
from SimulatedAnnealing import SA

L0 = 3.0128 * math.pow(10, 28)
L_sun = 3.828 * math.pow(10, 26)
R_sun = 695700000.0  # meter
M_sun = 1.98847 * math.pow(10, 30)
T_sun = 5778.0  # K
pc = 3.0857 * math.pow(10, 16)
au = 1.495978707 * math.pow(10, 11)
G = 6.67430 * math.pow(10, -11)


class RCMA(SA):
    """The class R CMa inherits from the class Annealer"""

    # pass extra data (the distance matrix) into the constructor
    def __init__(self, init_solution, phases_p, photometric_data, phases_rv1, phases_rv2, rv1, rv2):

        self.init_solution = init_solution
        self.phases_p = phases_p
        self.photometric_data = photometric_data
        self.p_max = max(photometric_data)
        self.rv1 = rv1
        self.rv2 = rv2
        self.phases_rv1 = phases_rv1
        self.phases_rv2 = phases_rv2

        # the center of the coordinate system is the gravity point of the R CMa system
        # the unit of the x-axis is the radius of the sun
        # the initial position is that the star 2 is in front of the star 1
        self.position1 = [-init_solution['star1']['orbital_radius'], 0]
        self.position2 = [init_solution['star2']['orbital_radius'], 0]

        self.position_view = [init_solution['distance'], 0]

        y_max = np.max(self.rv1)
        y_min = np.min(self.rv1)
        y_median = (y_max + y_min) / 2
        self.rv1 = self.rv1 - y_median

        y_max = np.max(self.rv2)
        y_min = np.min(self.rv2)
        y_median = (y_max + y_min) / 2
        self.rv2 = self.rv2 - y_median

        super(RCMA, self).__init__(init_solution)

    def move(self):
        """Change the radii of the orbits"""
        # hold that T = 2*pi*r/v same for both stars by 'self.ratio_r1_r2'
        self.curr_solution['orbit_coefficient_1_2'] = random.uniform(-0.01, 0.01) + self.curr_solution[
            'orbit_coefficient_1_2']

        while self.curr_solution['orbit_coefficient_1_2'] < 0:
            self.curr_solution['orbit_coefficient_1_2'] = random.uniform(0.0, 0.01) + self.curr_solution[
                'orbit_coefficient_1_2']

        while self.curr_solution['orbit_coefficient_1_2'] < 0:
            self.curr_solution['orbit_coefficient_1_2'] = random.uniform(0.0, 0.01) + self.curr_solution[
                'orbit_coefficient_1_2']

        self.curr_solution['mass_coefficient_1_2'] = 1 / self.curr_solution['orbit_coefficient_1_2']

    def move_2(self):
        """Change the radii of the stars"""
        movement = 0.01
        # np.random.seed()
        x = random.uniform(-movement, movement)
        y = random.uniform(-movement, movement)
        r1 = x + self.curr_solution['star1']['radius']
        r2 = y + self.curr_solution['star2']['radius']
        # print(x, y)

        while r1 < 0:
            r1 = random.uniform(0.0, movement) + r1

        while r2 < 0:
            r2 = random.uniform(0.0, movement) + r2

        if r1 < r2:
            # print(x, y)
            # print(r1, r2)
            return -1

        # print(22222222222)

        self.curr_solution['star1']['radius'] = r1
        self.curr_solution['star2']['radius'] = r2

    def move_3(self):
        """Change the temperature of the stars"""
        movement = 0.05
        self.curr_solution['star1']['temperature'] = random.uniform(-movement, movement) + self.curr_solution['star1'][
            'temperature']
        self.curr_solution['star2']['temperature'] = random.uniform(-movement, movement) + self.curr_solution['star2'][
            'temperature']

        while self.curr_solution['star1']['temperature'] < 0:
            self.curr_solution['star1']['temperature'] = random.uniform(0.0, movement) + self.curr_solution['star1'][
                'temperature']

        while self.curr_solution['star2']['temperature'] < 0:
            self.curr_solution['star2']['temperature'] = random.uniform(0.0, movement) + self.curr_solution['star2'][
                'temperature']

        self.curr_solution['star1']['luminosity'] = math.pow(self.curr_solution['star1']['radius'], 2) * math.pow(
            self.curr_solution['star1']['temperature'] / T_sun, 4)

        self.curr_solution['star2']['luminosity'] = math.pow(self.curr_solution['star2']['radius'], 2) * math.pow(
            self.curr_solution['star2']['temperature'] / T_sun, 4)

    def cal_distance(self):
        """Calculate the distance between two points in the coordinate system. The unit is pc(parsecs)"""
        d1 = math.pow((self.position1[0] * R_sun / pc - self.position_view[0] * R_sun / pc), 2)
        d2 = math.pow((self.position1[1] * R_sun / pc - self.position_view[1] * R_sun / pc), 2)

        d3 = math.pow((self.position2[0] * R_sun / pc - self.position_view[0] * R_sun / pc), 2)
        d4 = math.pow((self.position2[1] * R_sun / pc - self.position_view[1] * R_sun / pc), 2)
        return math.sqrt(d1 + d2), math.sqrt(d3 + d4)

    def shadow(self):
        """Takes a star, return an area that is obscured.
        Find two lines which are tangent to the circle and their projection to y-axis"""

        y_projection_1 = [self.position1[1] - self.curr_solution['star1']['radius'],
                          self.position1[1] + self.curr_solution['star1']['radius']]
        y_projection_2 = [self.position2[1] - self.curr_solution['star2']['radius'],
                          self.position2[1] + self.curr_solution['star2']['radius']]

        # calculate the common area of the two y projections
        # calculate the area of circular sector - the area of the triangle
        # known length is the vertical interception, which is the common part of the two y-projections
        flag = -1

        if self.position2[0] > self.position1[0]:  # the smaller star is in the front
            flag = 0
        elif self.position1[0] > self.position2[0]:  # the bigger star is in the front
            flag = 1
        else:
            return 0, flag

        if y_projection_2[1] < y_projection_1[0] or y_projection_2[0] > y_projection_1[1]:
            return 0, flag

        if y_projection_1[0] <= y_projection_2[0] and y_projection_1[1] >= y_projection_2[1]:
            return math.pi * math.pow(self.curr_solution['star2']['radius'], 2), flag

        distance1to2 = self.position2[1] - self.position1[1]
        r1 = self.curr_solution['star1']['radius']
        r2 = self.curr_solution['star2']['radius']

        if distance1to2 < 0:
            distance1to2 = -distance1to2

        # the distance from star 2 to the center of the intersection
        b = (math.pow(distance1to2, 2) - math.pow(r1, 2) + math.pow(r2, 2)) / 2 / distance1to2

        a = distance1to2 - b  # the distance from star 1 to the center of the intersection

        if math.pow(r1, 2) - math.pow(a, 2) < 0:
            print(r1, a, b, distance1to2)
            return 0, -1

        y = math.sqrt(math.pow(r1, 2) - math.pow(a, 2))

        angle1 = math.degrees(math.acos(a / r1)) * 2
        angle2 = math.degrees(math.acos(b / r2)) * 2
        area1 = math.pi * math.pow(r1, 2) * angle1 / 360 - a * y
        area2 = math.pi * math.pow(r2, 2) * angle2 / 360 - b * y
        return area1 + area2, flag

    def luminosity(self):
        distance1, distance2 = self.cal_distance()

        flux1 = self.curr_solution['star1']['luminosity'] * L_sun / 4 / math.pi / distance1 / distance1 / pc / pc
        flux2 = self.curr_solution['star2']['luminosity'] * L_sun / 4 / math.pi / distance2 / distance2 / pc / pc

        # M1 = -2.5 * math.log10(self.curr_solution['star1']['luminosity'] * L_sun / L0)  # parameter
        # m1 = M1 - 5 + 5 * math.log10(distance1)
        #
        # M2 = -2.5 * math.log10(self.curr_solution['star2']['luminosity'] * L_sun / L0)  # parameter
        # m2 = M2 - 5 + 5 * math.log10(distance2)

        # print(m1, m2)
        m1 = flux1
        m2 = flux2

        # calculate the shadow
        s3, flag = self.shadow()

        if flag == 0:  # the smaller star is in the front, star2
            s2 = math.pi * math.pow(self.curr_solution['star2']['radius'], 2)
            if s3 > s2:
                s3 = s2
            m1 = m1 - m1 * s3 / s2

        elif flag == 1:  # the larger star 1 is in the front
            s2 = math.pi * math.pow(self.curr_solution['star2']['radius'], 2)
            if s3 > s2:
                s3 = s2
            m2 = m2 - m2 * s3 / s2

        # print(m1, m2)
        return m1 + m2

    # objective function
    def evaluation(self, show=False):
        """Compare the result with real data"""
        # phases = list()
        rv1 = list()
        rv2 = list()
        predictions = list()
        positions_1 = list()
        positions_2 = list()

        v1 = math.sqrt(
            G * self.curr_solution['star2']['mass'] * self.curr_solution['mass_coefficient_1_2'] * M_sun *
            self.curr_solution['star1']['orbital_radius'] * self.curr_solution[
                'orbit_coefficient_1_2'] * R_sun / math.pow(
                (self.curr_solution['star1']['orbital_radius'] + self.curr_solution['star2']['orbital_radius']) *
                self.curr_solution['orbit_coefficient_1_2'] * R_sun, 2)) / 1000.0

        v2 = math.sqrt(
            G * self.curr_solution['star1']['mass'] * self.curr_solution['mass_coefficient_1_2'] * M_sun *
            self.curr_solution['star2']['orbital_radius'] * self.curr_solution[
                'orbit_coefficient_1_2'] * R_sun / math.pow(
                (self.curr_solution['star1']['orbital_radius'] + self.curr_solution['star2']['orbital_radius']) *
                self.curr_solution['orbit_coefficient_1_2'] * R_sun, 2)) / 1000.0

        # print(v1, v2)

        # steps = 36000
        # for i in [0, 1]:
        for index in self.phases_p:
            # build a 2d coordinate system, the origin is at the center of the two orbits
            # the start of a phase is: m2 is in front of m1
            self.position1 = [
                -self.curr_solution['star1']['orbital_radius'] * math.cos(2 * math.pi * index) * self.curr_solution[
                    'orbit_coefficient_1_2'],
                self.curr_solution['star1']['orbital_radius'] * math.sin(2 * math.pi * index) * self.curr_solution[
                    'orbit_coefficient_1_2']]
            self.position2 = [
                self.curr_solution['star2']['orbital_radius'] * math.cos(-2 * math.pi * index) * self.curr_solution[
                    'orbit_coefficient_1_2'],
                self.curr_solution['star2']['orbital_radius'] * math.sin(-2 * math.pi * index) * self.curr_solution[
                    'orbit_coefficient_1_2']]

            rv1.append(-v1 * math.sin(2 * math.pi * index))
            rv2.append(v2 * math.sin(2 * math.pi * index))

            prediction = self.luminosity()
            # phases.append(index / steps + i)
            predictions.append(prediction)

            positions_1.append(self.position1)
            positions_2.append(self.position2)

        # drawAnimation(positions_1, positions_2)

        # calculate the F0
        # F0 = L0 / 4 / math.pi / 44 / 44

        max_p = sum(self.photometric_data[
                        int(0.05 * len(self.photometric_data)): int(0.4 * len(self.photometric_data))]) / (int(
            0.4 * len(self.photometric_data)) - int(0.05 * len(self.photometric_data)))
        max_min_p = max_p - min(self.photometric_data)
        max_min_pre = max(predictions) - min(predictions)

        F0 = max_min_pre / max_min_p
        predictions = [predict / F0 for predict in predictions]

        prediction_avg_max = max(predictions)
        F0 = prediction_avg_max - max_p
        predictions = [predict - F0 for predict in predictions]
        diffs = predictions - self.photometric_data
        diffs = [math.pow(diff, 2) for diff in diffs]

        if show:
            # draw_compare(self.phases_p, predictions, self.photometric_data, 'Phase', 'Prediction')
            # draw_RVs_compare(self.phases_rv1, self.phases_rv2, self.phases_p, self.rv1, self.rv2, rv1, rv2,
            #                  'Phase', 'Radial Velocity')
            draw_RVs_p_compare(self.phases_rv1, self.phases_rv2, self.phases_p, self.rv1, self.rv2, rv1, rv2, 'Phase',
                               'Radial Velocity', self.phases_p, predictions, self.photometric_data, 'Phase',
                               'Flux')
            # print(max(predictions) / min(predictions), max(self.photometric_data) / min(self.photometric_data))

        return sum(diffs)
