"""
Copyright (c) 2015 Andreea Georgescu

Created on Wed Nov 19 00:18:55 2014

This program is free software: you can redistribute it and/or modify
it under the terms of the GNU General Public License as published by
the Free Software Foundation, either version 2 of the License, or
(at your option) any later version.

This program is distributed in the hope that it will be useful,
but WITHOUT ANY WARRANTY; without even the implied warranty of
MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
GNU General Public License for more details.

You should have received a copy of the GNU General Public License
along with this program.  If not, see <http://www.gnu.org/licenses/>.
"""

from __future__ import absolute_import
from __future__ import division
import numpy as np
from interp import interp1d
pi = np.pi

name = "LUX2013zero"
modulated = False

energy_resolution_type = "Poisson"


def EnergyResolution(e):
    return 0.37 * np.ones_like(e)

FFSD = 'GaussianFFSD'
FFSI = 'HelmFF'
FF = {'SI': FFSI,
      'SDPS': FFSD,
      'SDAV': FFSD,
      }
target_nuclide_AZC_list = \
    np.array([[124, 54, 0.0008966], [126, 54, 0.0008535], [128, 54, 0.018607],
              [129, 54, 0.25920], [130, 54, 0.040280], [131, 54, 0.21170],
              [132, 54, 0.27035], [134, 54, 0.10644], [136, 54, 0.09168]])
target_nuclide_JSpSn_list = \
    np.array([[0, 0, 0], [0, 0, 0], [0, 0, 0],
              [1./2, 0.010 * np.sqrt(3./2 / pi), .329 * np.sqrt(3./2 / pi)], [0, 0, 0],
              [3./2, -0.009 * np.sqrt(5./2 / pi), -.272 * np.sqrt(5./2 / pi)], [0, 0, 0],
              [0, 0, 0], [0, 0, 0]])
target_nuclide_mass_list = np.array([115.418, 117.279, 119.141, 120.074, 121.004,
                                     121.937, 122.868, 124.732, 126.597])
num_target_nuclides = target_nuclide_mass_list.size

QuenchingFactor = \
    interp1d(np.array([0, 3, 6, 9, 12, 15, 18, 21, 24, 27, 30, 33, 36, 40, 1000]),
             np.array([0, 0.394922, 0.714621, 0.861934, 0.944993, 1.01551, 1.08134,
                       1.13507, 1.19417, 1.23701, 1.28068, 1.31812, 1.35872, 1.4, 1.4]))

Ethreshold = 2.
Emaximum = 30.
ERmaximum = 36.

def Efficiency_ER(er):
    try:
        len(er)
    except TypeError:
        er = [er]
    return np.array([1. if e >= 3. else 0. for e in er])

Efficiency_interp = \
    interp1d(np.array([0.700414, 0.768683, 0.830262, 0.884938, 0.960924, 1.01607,
                       1.09747, 1.15124, 1.2238, 1.28035, 1.34308, 1.3977, 1.4584,
                       1.50166, 1.56273, 1.61336, 1.66564, 1.7105, 1.77062, 1.81347,
                       1.87722, 1.92777, 1.99023, 2.07117, 2.16112, 2.23115, 2.27304,
                       2.35294, 2.42273, 2.55497, 2.68015]),
             np.array([0., 0.005714, 0.015238, 0.030476, 0.049524, 0.068571, 0.099048,
                       0.127619, 0.167619, 0.2, 0.249524, 0.285714, 0.340952, 0.380952,
                       0.434286, 0.47619, 0.527619, 0.573333, 0.628571, 0.668571,
                       0.727619, 0.767619, 0.817143, 0.860952, 0.910476, 0.933333,
                       0.948571, 0.967619, 0.980952, 0.994286, 1.]))

def Efficiency(e):
    return Efficiency_interp(e) if 0.700414 <= e < 2.68015 \
            else np.array(0.) if e < 0.700414 else np.array(1.)

Exposure = 85.3 * 118.3
ERecoilList = np.array([])

BinSize = 3.5
BinEdges_left = np.array([2.0, 5.5, 9.0, 12.5, 16.0, 19.5, 23.0, 26.5])
BinEdges_right = np.array([5.5, 9.0, 12.5, 16.0, 19.5, 23.0, 26.5, 30])
BinBkgr = [0.03, 0.04, 0.06, 0.07, 0.09, .10, 0.12, 0.13]
BinData = np.array([0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0])
BinError = np.array([0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0])
