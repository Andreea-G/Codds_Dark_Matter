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
from __future__ import print_function

import numpy as np
import os
from interp import interp1d
from globalfnc import ConfidenceLevel, chi_squared1
pi = np.pi

name = "PICO_60"
modulated = False

energy_resolution_type = "Dirac"
# actually Bubble Nucleation, but similar enough to implement like Dirac

eff = np.loadtxt(os.getcwd() + '/Data/pico60_eff.dat')
p_eff = interp1d(eff[:,0],eff[:,1], kind='linear')

def EnergyResolution(e):
    return 0.1 * np.ones_like(e)

FFSD = 'GaussianFFSD'
FFSI = 'HelmFF'
FF = {'SI': FFSI,
      'SDPS': FFSD,
      'SDAV': FFSD,
      }
target_nuclide_AZC_list = np.array([[19, 9, 0.80]])
target_nuclide_JSpSn_list = \
    np.array([[1./2, 0.4751 * np.sqrt(3./2 / pi), -0.0087 * np.sqrt(3./2 / pi)]])
target_nuclide_mass_list = np.array([17.6969])
num_target_nuclides = target_nuclide_mass_list.size

def QuenchingFactor(e):
    return np.ones_like(e)

Ethreshold = 6.
Emaximum = 100
ERmaximum = np.inf

def Efficiency_ER(er):
    #if er >= eff[0,0] and er < eff[-1,0]:
    #    return p_eff(er)
    #else:
    #    return np.ones_like(er)
    return np.ones_like(er)

alpha = 5.
def Efficiency(e, er):
    #return 1. - np.exp(alpha * (1. - er/e))
    if er > Ethreshold and er < Emaximum:
        return 1.
    else:
        return 0.

Exposure = 1167.
ERecoilList = np.array([])

