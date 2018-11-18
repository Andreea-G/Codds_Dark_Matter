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
    
    
    Results from arXiv 1607.07400:::: Updated to 1708.06917
    """

from __future__ import absolute_import
from __future__ import division
import numpy as np
from scipy.interpolate import interp1d
pi = np.pi

name = "PandaX"
modulated = False

#energy_resolution_type = "Poisson"
energy_resolution_type = "Gaussian"


def EnergyResolution(e):
    return .1 * np.ones_like(e)

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




def QuenchingFactor(e_list):
    """
    Ly = 2.28
    Snr = 0.95
    See = 0.58
    try:
        len(e_list)
    except TypeError:
        e_list = [e_list]
    
    q = np.array([0. if e < 1 \
                  else  QuenchingFactor_interp(np.log10(e)) if e < 10**2.23187 \
                  else 0.2
                  for e in e_list])
    return Ly * Snr / See * q
    """
    try:
        len(e_list)
    except TypeError:
        e_list = [e_list]
    return np.ones_like(e_list)




Ethreshold = 0.
Emaximum = 45.
ERmaximum = 35.

Eff_ER_interp = \
    interp1d(np.loadtxt('/Users/SamWitte/Desktop/Codds_DarkMatter/src/Data/eff_Pandax2016.dat')[:,0],
             np.loadtxt('/Users/SamWitte/Desktop/Codds_DarkMatter/src/Data/eff_Pandax2016.dat')[:,1],
             kind='linear', bounds_error=False,fill_value=0.)

def Efficiency_ER(er):
    try:
        len(er)
    except TypeError:
        er = [er]
    return np.array([Eff_ER_interp(e) if e > 1. and e < 35. else 0. for e in er])


def Efficiency(e):
    try:
        len(e)
    except TypeError:
        e = [e]
    return np.array([1. if ee > 0. and ee < 40. else 0. for ee in e])
#return Efficiency_interp(e) if 0.700414 <= e < 2.68015 \
#   else np.array(0.) if e < 0.700414 else np.array(1.)

Exposure = 5.4 * 10 ** 4.
ERecoilList = np.array([28.])


BinData = np.array([1.])
BinEdges_left = np.array([1.1])
BinEdges_right = np.array([35.])
BinBkgr=np.array([5.])
BinSize=80.0
BinExposure=np.array([Exposure])



