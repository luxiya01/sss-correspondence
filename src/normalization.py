from dataclasses import dataclass
import math
from scipy.special import j1
import numpy as np
import matplotlib.pyplot as plt


@dataclass
class SSSHit:
    """Class representing one sidescan sonar hit on the seafloor.
    Unless specified, the ground_height is defaulted to 0 m, i.e. flat seafloor assumption
    is used.

    Parameters
    ----------
    r_s: float
        Slant range.
    h: float
        Vehicle height.
    h_s: float
        Height of the hit point above the seafloor, default to 0m.
    """
    r_s: float
    h: float
    h_s: float = 0

    @property
    def grazing_angle(self):
        """Returns the grazing angle of the SSSHit. Using Eq(1) from 
        `On-Line Multi-Class Segmentation of Side-Scan Sonar Imagery Using an Autonomous
        Unverwater  Vehicle`
        """
        return math.asin((self.h - self.h_s) / self.r_s)


def estimate_transducer_radius(wavelength: float,
                               vertical_opening: float) -> float:
    """Estimate transducer radius using Eq(8) from
    `On-Line Multi-Class Segmentation of Side-Scan Sonar Imagery Using an Autonomous
    Unverwater  Vehicle`

    Parameters
    ----------
    wavelength : float
        Wavelength of the transmitted sound wave.
    vertical_opening : float
        Vertical opening of the side-scan sonar sensor. Typically 10s of degrees.

    Returns
    -------
    radius: float
        Estimated transducer radius.
    """
    radius = (3.8317 * wavelength) / (2 * math.pi *
                                      math.sin(vertical_opening / 2))
    return radius


def echo_intensity_at_ssshit(hit: SSSHit, frequency: float, wavelength: float,
                             mounting_angle: float,
                             transducer_radius: float) -> float:
    """Estimate echo intensity at a SSS hit on the seafloor using Eq(5) from 
    `On-Line Multi-Class Segmentation of Side-Scan Sonar Imagery Using an Autonomous
    Unverwater  Vehicle`

    Parameters
    ----------
    hit : SSSHit
        The point where the echo intensity is to be estimated.
    frequency: float
        The emitted wave frequency.
    wavelength: float
        The emitted wavelength.
    mounting_angle : float
        Mounting angle of the SSS.
    transducer_radius : float
        Radius of the SSS, can be estimated using `estimate_transducer_radius` function.

    Returns
    -------
    intensity: float
        Modelled echo intensity value.
    """
    angle_diff = hit.grazing_angle - mounting_angle

    # Computing the term in the parenthesis^2
    denom = 2 * math.pi / wavelength * transducer_radius * math.sin(angle_diff)
    bessel_of_denom = j1(denom)
    paren_term = math.pow(2 * bessel_of_denom / denom, 2)

    intensity = frequency * math.pow(transducer_radius, 4) / math.pow(
        hit.r_s, 2) * paren_term
    return intensity