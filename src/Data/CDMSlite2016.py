"""

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
pi = np.pi

name = "CDMSlite2016"
modulated = False

energy_resolution_type = "Gaussian"

def EnergyResolution(e):
    return 0.014 * np.ones_like(e)

FFSD = 'GaussianFFSD'
FFSI = 'HelmFF'
FF = {'SI': FFSI,
      'SDPS': FFSD,
      'SDAV': FFSD,
      }

target_nuclide_AZC_list = np.array([[70., 32., 0.19608], [72., 32., 0.27040],
                                    [73., 32., 0.07790], [74., 32., 0.37378],
                                    [76., 32., 0.08184]])
target_nuclide_JSpSn_list = \
    np.array([[0., 0., 0.], [0., 0., 0.],
              [9./2, 0.0392517 * np.sqrt(((2*9./2 + 1)*(9./2 + 1))/(4*pi*9./2)),
               .375312 * np.sqrt(((2*9./2 + 1)*(9./2 + 1))/(4*pi*9./2))],
              [0., 0., 0.], [0., 0., 0.]])
target_nuclide_mass_list = np.array([65.134, 66.995, 67.9278, 68.8571, 70.7203])

num_target_nuclides = target_nuclide_mass_list.size

def QuenchingFactor(e):
    return (1 + 69./3 * 0.19935 * e**0.1204)/(1 + 69./3)

Ethreshold = 0.36
Emaximum = 1.04
ERmaximum = 7

def Efficiency(e): return np.array(.55) if Ethreshold <= e < Emaximum else np.array(0.)

def Efficiency_ER(er):
    return np.ones_like(er)

Exposure = 70.1
ERecoilList = np.array([0.41, 0.45, 0.49, 0.52, 0.54, 0.61, 0.64, 0.66, 0.70, 0.73, 0.78, 0.84])
