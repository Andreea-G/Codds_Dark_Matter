"""

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

## SODIUM ONLY

from __future__ import absolute_import
from __future__ import division
import numpy as np
pi = np.pi

name = "DAMA_2Bin"
modulated = True

energy_resolution_type = "Gaussian"

def EnergyResolution(e):
    return 0.448 * np.sqrt(e) + 0.0091 * e

FFSD = 'GaussianFFSD'
FFSI = 'HelmFF'
FF = {'SI': FFSI,
      'SDPS': FFSD,
      'SDAV': FFSD,
      }

target_nuclide_AZC_list = np.array([[23, 11, 0.153373]])
target_nuclide_JSpSn_list = \
    np.array([[3./2, 0.2477 * np.sqrt(5./3 / pi), .0198 * np.sqrt(5./3 / pi)]])
target_nuclide_mass_list = np.array([21.4148])
num_target_nuclides = target_nuclide_mass_list.size

def QuenchingFactor(e):
    return 0.4 * np.ones_like(e)

def QuenchingFactorOfEee(e):
    return QuenchingFactor(e)  # since it's a constant function

Ethreshold = 2.
Emaximum = 1000.
ERmaximum = 2500.

def Efficiency_ER(er):
    return np.ones_like(er)

def Efficiency(e):
    return np.array(1.) if Ethreshold <= e < Emaximum else np.array(0.)

Exposure = 1.33 * 1000 * 365.25
ERecoilList = np.array([])

BinEdges = np.array([2., 6., 14.])
BinData = np.array([0.0106 * 10.0, 0.0001 * 20.0])  # Table 6 of 1308.5109 in units of counts / (kg * day)
BinError = np.array([0.0012 * 10.0, 0.0007 * 20.0])
