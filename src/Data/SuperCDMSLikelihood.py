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
from globalfnc import ConfidenceLevel, chi_squared1
pi = np.pi

name = "SuperCDMS"
modulated = False

energy_resolution_type = "Dirac"


def EnergyResolution(e):
    return np.ones_like(e)

FFSD = 'GaussianFFSD'
FFSI = 'HelmFF'
FF = {'SI': FFSI,
      'SDPS': FFSD,
      'SDAV': FFSD,
      }
target_nuclide_AZC_list = \
    np.array([[70., 32., 0.19608], [72., 32., 0.27040], [73., 32., 0.07790],
              [74., 32., 0.37378], [76., 32., 0.08184]])
target_nuclide_JSpSn_list = \
    np.array([[0., 0., 0.], [0., 0., 0.],
              [9./2, 0.0392517 * np.sqrt(((2*9./2 + 1)*(9./2 + 1))/(4*pi*9./2)),
               .375312 * np.sqrt(((2*9./2 + 1)*(9./2 + 1))/(4*pi*9./2))],
              [0., 0., 0.], [0., 0., 0.]])
target_nuclide_mass_list = np.array([65.134, 66.995, 67.9278, 68.8571, 70.7203])

num_target_nuclides = target_nuclide_mass_list.size


def QuenchingFactor(e):
    return np.ones_like(e)

Ethreshold = 1.63799
Emaximum = 10.0011
ERmaximum = 10.0011

Efficiency_interp = \
    interp1d(np.array([1.63799, 1.93525, 2.35928, 2.37871, 3.12938, 3.15831,
                       3.8895, 3.90877, 4.2841, 4.30358, 4.63016, 4.64942, 5.38539,
                       5.4095, 5.78968, 6.15036, 6.16481, 6.8911, 6.92511, 9.16257,
                       9.18213, 10.0011]),
             np.array([0.044225, 0.071339, 0.086737, 0.105692, 0.112107, 0.196045,
                       0.19975, 0.260222, 0.26388, 0.268395, 0.275658, 0.339739, 0.366008,
                       0.43731, 0.44819, 0.459066, 0.506, 0.514216, 0.543101, 0.544292,
                       0.529854, 0.532668]))


def Efficiency(e, er):
    return np.ones_like(er)


def Efficiency_ER(e):
    return Efficiency_interp(e) if Ethreshold <= e < Emaximum else np.array(0.)

Exposure = 577.
ERecoilList = np.array([1.7, 1.8, 1.9, 1.9, 2.3, 2.7, 3.0, 5.8, 7.0, 7.8, 9.4])

# BinSize = 1.05
# BinEdges_left = np.array([1.6, 2.65, 3.7, 4.75, 5.8, 6.85, 7.9, 8.95])
# BinEdges_right = np.array([2.65, 3.7, 4.75, 5.8, 6.85, 7.9, 8.95, 10.0])
# BinData = np.array([5, 2, 0, 1, 0, 2, 0, 1])
# BinError = np.array([1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0])
# BinBkgr = []

# Expected_limit = BinData + np.sqrt(chi_squared1(ConfidenceLevel)) * BinError # Testing limit

BinSize = 8.4
BinEdges_left = np.array([1.64])
BinEdges_right = np.array([10.0])
BinData = np.array([11])
BinError = np.array([1.0])
BinBkgr = np.array([6.56])  # +/- (1.39, 1.17)
BinExposure = np.array([577.])
Expected_limit = 10.13
