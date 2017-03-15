"""
    Copyright (c) 2017 Sam Witte
    
    Created on Jan 19, 2017
    
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
    
    
    Results from
    """

from __future__ import absolute_import
from __future__ import division
import numpy as np
from scipy.interpolate import interp1d
pi = np.pi

name = "DarkSideG2"
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
    np.array([[40, 18, 1.]])
target_nuclide_JSpSn_list = \
    np.array([[0., 0., 0.]])
target_nuclide_mass_list = np.array([37.42])
num_target_nuclides = target_nuclide_mass_list.size



Ethreshold = 40.
Emaximum = 240.
ERmaximum = 240.

def QuenchingFactor(e):
    return np.ones_like(e)

def Efficiency_ER(er):
    try:
        len(er)
    except TypeError:
        er = [er]
    return np.ones_like(er)


def Efficiency(e,er):
    try:
        len(er)
    except TypeError:
        er = [er]
    return 0.7 * np.array([1.0 if ee > Ethreshold and ee < ERmaximum else 0. for ee in er])



Exposure = 20. * 1000. * 365.24 * 3.
ERecoilList = np.array([])


