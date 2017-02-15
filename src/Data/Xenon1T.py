"""
Copyright (c) 2015 Andreea Georgescu

Created on Tue Apr  7 17:25:46 2015

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
from scipy.interpolate import interp1d
pi = np.pi

name = "Xenon1T"
modulated = False

energy_resolution_type = "Poisson"
#energy_resolution_type = "Gaussian"

def EnergyResolution(e):
    return 0.5 * np.ones_like(e)
#    return np.ones_like(e)

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
              [3./2, -0.009 * np.sqrt(5./2 / pi), -.272 * np.sqrt(5./2 / pi)],
              [0, 0, 0], [0, 0, 0], [0, 0, 0]])
target_nuclide_mass_list = np.array([115.418, 117.279, 119.141, 120.074, 121.004,
                                     121.937, 122.868, 124.732, 126.597])
num_target_nuclides = target_nuclide_mass_list.size

xq = np.array([0., 0.395837, 0.431065, 0.451241, 0.473339, 0.582546, 0.7747, 0.995997,
              1.09303, 1.18078, 1.30024, 1.41681, 1.4783, 1.6948, 1.76781, 1.9004,
              2.01954, 2.14444, 2.23187 ])
yq = np.array([0., 0.0745321, 0.0792141, 0.0831369, 0.0941459, 0.0999667, 0.116417,
              0.144003, 0.147293, 0.154252, 0.159694, 0.169311, 0.17665, 0.191076,
              0.194872, 0.198035, 0.197403, 0.2, 0.2])

QuenchingFactor_interp = interp1d(xq, yq, kind='linear', bounds_error=False, fill_value=0.)

def QuenchingFactor(e_list):
    Ly = 2.28
    Snr = 0.95
    See = 0.58
    try:
        len(e_list)
    except TypeError:
        e_list = [e_list]
    
    q = np.array([0. if e < 1 \
                  else  QuenchingFactor_interp(np.log10(e)) if np.log10(e) < 2.23 \
                  else 0.2
                  for e in e_list])
    return Ly * Snr / See * q


Ethreshold = 3
Emaximum = 70
ERmaximum = 30.



def Efficiency(e):
    return 0. if e < Ethreshold else 1.0 if e < Emaximum else 1.



def Efficiency_ER(er):
    try:
        len(er)
    except TypeError:
        er = [er]
    return np.array([0.4 for e in er])


Exposure = 1000. * 365. * 2.
ERecoilList = np.array([])
