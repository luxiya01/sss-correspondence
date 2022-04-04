"""
This module contains simple shapes used to split the sss_meas_data into training and test segments.

Exposed data classes:
    - Point2D
    - Line2D

Exposed functions:
    - get_intercept(line1: Line2D, line2: Line2D):
    - get_intercept_between_line1_normal_at_given_point_and_line2(line1: Line2D, point: Point2D,
      line2: Line2D)
    - get_intercepts_between_line1_normal_and_line2(line1: Line2D, line2: Line2D)
"""

from dataclasses import dataclass
from matplotlib import pyplot as plt
import numpy as np
import seaborn as sns


@dataclass
class Point2D:
    """A 2D point defined by x and y values."""
    x: float
    y: float

    @classmethod
    def from_array(cls, array):
        return cls(x=array[0], y=array[1])

    def __repr__(self):
        return f'({self.x:.2}, {self.y:.2})'


@dataclass
class Line2D:
    """A 2D line segment defined by two points.

    point1: Point2D
        Starting point of the line segment
    point2: Point2D
        Ending point of the line segment
    """
    point1: Point2D
    point2: Point2D

    @property
    def dx(self):
        return self.point2.x - self.point1.x

    @property
    def dy(self):
        return self.point2.y - self.point1.y

    @property
    def slope(self):
        return self.dy / self.dx

    @property
    def normal(self):
        return -self.dx / self.dy

    @property
    def intercept(self):
        return self.point1.y - self.slope * self.point1.x

    def normal_intercept(self, point: Point2D):
        return point.y - self.normal * point.x

    def plot(self):
        with sns.color_palette(n_colors=3):
            fig, ax = plt.subplots()

            ax.scatter([self.point1.x, self.point2.x],
                       [self.point1.y, self.point2.y],
                       marker='o')

            x_min = min(self.point1.x, self.point2.x)
            x_max = max(self.point1.x, self.point2.x)
            x_vals = np.linspace(x_min, x_max, int(x_max - x_min) * 50)

            ax.plot(x_vals, x_vals * self.slope + self.intercept, label='line')
            ax.plot(x_vals,
                    x_vals * self.normal + self.normal_intercept(self.point1),
                    label='normal passing point1')
            ax.plot(x_vals,
                    x_vals * self.normal + self.normal_intercept(self.point2),
                    label='normal passing point2')

            ax.axis('equal')
            ax.legend()
            ax.set_title(
                f'Normals and line defined by points {self.point1} and {self.point2}'
            )


def get_intercept(line1: Line2D, line2: Line2D):
    """Returns the Point2D intercept between two lines"""
    x = (line2.intercept - line1.intercept) / (line1.slope - line2.slope)
    y = line1.slope * x + line1.intercept
    return Point2D(x, y)


def get_intercept_between_line1_normal_at_given_point_and_line2(
        line1: Line2D, point: Point2D, line2: Line2D):
    """Returns the Point2D intercept between (line1's normal defined at a given point) and (line2)"""
    normal_intercept = line1.normal_intercept(point)
    x = (line2.intercept - normal_intercept) / (line1.normal - line2.slope)
    y = line1.normal * x + normal_intercept
    return Point2D(x, y)


def get_intercepts_between_line1_normal_and_line2(line1: Line2D,
                                                  line2: Line2D):
    """Returns two Point2D intercepts between line1's normal and line2, the two intercepts
    represents the intercept with line2 from line1's normal defined using line1's internal point1
    (starting point) and internal point2 (ending point), respectively."""
    p1 = get_intercept_between_line1_normal_at_given_point_and_line2(
        line1, line1.point1, line2)
    p2 = get_intercept_between_line1_normal_at_given_point_and_line2(
        line1, line1.point2, line2)
    return p1, p2
