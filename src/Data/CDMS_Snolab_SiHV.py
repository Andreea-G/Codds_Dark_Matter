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
from scipy.interpolate import interp1d
pi = np.pi

name = "CDMS_Snolab_SiHV"
modulated = False

energy_resolution_type = "Dirac"

def EnergyResolution(e):
    return 0.1 * np.ones_like(e)

FFSD = 'GaussianFFSD'
FFSI = 'HelmFF'
FF = {'SI': FFSI,
      'SDPS': FFSD,
      'SDAV': FFSD,
      }

target_nuclide_AZC_list = np.array([[28, 14, 0.918663943428171], [29, 14, 0.04833558589888038],
                                    [30, 14, 0.03300047067294847]])
target_nuclide_JSpSn_list = np.array([[0, 0, 0], [1./2, -0.0019 * np.sqrt(3./(2 * pi)),
                                                  .1334 * np.sqrt(3./(2 * pi))], [0, 0, 0]])
target_nuclide_mass_list = np.array([26.0603, 26.9914, 27.9204])
num_target_nuclides = target_nuclide_mass_list.size

ionyload = np.loadtxt('/Users/SamWitte/Desktop/Codds_DarkMatter/src/Data/si_ion.dat')
iony = interp1d(ionyload[:,0], ionyload[:,1], kind='linear',
                      fill_value='extrapolate', bounds_error=False)

def QuenchingFactor(e):
    return (iony(e) * 100. / 3.82 + 1.)

Ethreshold = 0.04
Emaximum = 100.0
ERmaximum = 2.

#def Efficiency(e): return np.array(1.0) if Ethreshold <= e < Emaximum else np.array(0.)

def Efficiency(e,er):
    try:
        len(er)
    except TypeError:
        er = [er]
    return np.array([1. if ee > Ethreshold and ee < ERmaximum else 0. for ee in er])

def Efficiency_ER(er):
    return np.ones_like(er)


Exposure = 9.6 * 4.
ERecoilList = np.array([])
