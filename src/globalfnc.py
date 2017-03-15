"""
Copyright (c) 2015 Andreea Georgescu

Created on Fri Nov 21 02:49:53 2014

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

from __future__ import division
from __future__ import absolute_import
from __future__ import print_function
import numpy as np
import math
from collections import OrderedDict
from scipy.special import erf, erfinv
from scipy.stats import chi2
pi = np.pi

T = True    # short-hand notation
F = False

PRECISSION = 1.e-5
# Unit conversions
fermiGeV = 1./0.1973269602  # Natural[GeV femto Meter]
kilogram = 1e-9/1.782661758e-36  # kg in GeV  (units: [GeV/kg])
SpeedOfLight = 299792.458  # km/s
AtomicMassUnit = 0.931494028
ProtonMass = 1.00727646677 * AtomicMassUnit
mPhiRef = 1000.
rho = 0.3  # GeV/cm**3
conversion_factor = rho * SpeedOfLight**2 * 1e5 * 3600 * 24
ConfidenceLevel = 0.9

v0bar = default_v0bar = 220.
vobs = default_vobs = default_v0bar + 12.
vesc = default_vesc = 533.

if False:   # alternative velocities; kept for reference
    v0bar = 230 - 3 * 24.4
    vobs = v0bar + 12
    vesc = 544 - 3 * 39

""" List of experiment names corresponding to each type of statistical analysis.
"""
MaximumGapLimit_exper = ["SuperCDMS",
                         "LUX2013zero", "LUX2013one", "LUX2013three", "LUX2013five", "LUX2013many",
                         "XENON10", "XENON100", "CDMSlite2013CoGeNTQ",
                         "LUX2016one", "LUX2016five", "LUX2016many","PandaX","CDMSlite2016", "Xenon1T",
                         "CDMS_Snolab_GeHV", "LZ","Darwin", "CDMS_Snolab_SiHV","CDMS_Snolab_GeiZip",
                         "PICO_500", "DarkSideG2","PICO_60"]
GaussianLimit_exper = ["KIMS2012", "PICASSO"]
BinnedSignal_exper = ["DAMA2010Na", "DAMA2010I"]
Crosses_exper = ["CDMSSi2012", "DAMA2010Na", "DAMA2010I"]
DAMALimit_exper = ["DAMA2010Na_TotRateLimit"]
Poisson_exper = ["SIMPLEModeStage2"]
EHImethod_exper = ["CDMSSi2012", "CDMSSiGeArtif", "CDMSSiArtif"]
SHM_line = ["SHM_eta0", "SHM_eta1"]
Extended_Like = ["CDMSSi2012"]
Poisson_Like = ["SuperCDMSLessT5", "SuperCDMSLikelihood","LUX2016zero"]
Poisson_likelihood = ["SuperCDMSLessT5", "SuperCDMSLikelihood", "LUX2016zero"]

""" Colors for plotting.
"""
Color = {"SuperCDMS": 'peru',
         "LUX2013zero": 'magenta', "LUX2013one": 'magenta', "LUX2013three": 'magenta',
         "LUX2013five": 'magenta', "LUX2013many": 'magenta',
         "XENON10": 'orange', "XENON100": 'royalblue',
         "CDMSlite2013CoGeNTQ": 'cyan', "CDMSSi2012": 'red',
         "CDMSSi2012_EHI": 'firebrick',
         "KIMS2012": 'purple', "PICASSO": 'darkturquoise',
         "DAMA2010Na_TotRateLimit": 'black',
         "DAMA2010Na": 'green',
         "DAMA2010I": 'green',
         "SIMPLEModeStage2": 'saddlebrown',
         "SHM_eta0": 'gray', "SHM_eta1": 'gray', "SuperCDMSLessT5": 'peru',
         "SuperCDMSLikelihood": 'peru',
         "LUX2016zero": 'navy', "LUX2016five": 'navy', "LUX2016many": 'navy', "PandaX":'darkorchid',
         "LUX2016one": 'navy', "CDMSlite2016": 'green', "Xenon1T": 'royalblue',
         "CDMS_Snolab_GeHV": 'darkturquoise',
         "LZ": 'black',"Darwin":'black', "CDMS_Snolab_SiHV":'darkturquoise',
         "CDMS_Snolab_GeiZip":'darkturquoise', "PICO_500":'black', "DarkSideG2":'black',"PICO_60":'green'}

""" Linestyles get cicled through for each experiment name.
"""
linestyles = ['-', '--', '-.', ':']
""" For some experiments the linestyles are fixed and customized, passed as dashes.
"""
line_dashes = {"LUX2013one": (8, 4, 3, 4, 3, 4),
               "LUX2013three": (8, 4, 3, 4), "LUX2013five": (8, 4), "LUX2013many": None,
               "SHM_eta0": (8, 4), "SHM_eta1": (3, 4), "SuperCDMSLessT5": (3, 4),
               "SuperCDMSLikelihood": (8, 4, 3, 4, 3, 4),
               "LUX2016five": (8, 4, 3, 4, 3, 4),
               "LUX2016many": (8, 4, 3, 4),
               "LUX2016one": (3,4) }
""" Legend names, in order of appearence in the legend for the corresponding experiments
that appear in the plot.
"""
legend_names = OrderedDict([("DAMA$_0", ["DAMA2010Na_TotRateLimit"]),
                            ("DAMA$_1$", ["DAMA2010Na", "DAMA2010I",
                                          "DAMA2010Na DAMA2010I",
                                          "DAMA2010I DAMA2010Na"]),
                            ("CDMS-II-Si", ["CDMSSi2012", "CDMSSi2012_EHI"]),
                            ("SuperCDMS", ["SuperCDMS"]),
                            ("CDMSlite", ["CDMSlite2013CoGeNTQ"]),
                            ("SIMPLE", ["SIMPLEModeStage2"]),
                            ("XENON10", ["XENON10"]), ("XENON100", ["XENON100"]),
                             ("Xenon1T", ["Xenon1T"]),
                            ("LUX2013", ["LUX2013zero", "LUX2013one", "LUX2013three",
                                     "LUX2013five", "LUX2013many"]),
                            ("PICASSO", ["PICASSO"]), ("KIMS", ["KIMS2012"]),
                            ("SHM $(\sigma_p = 10^{-40}\mathrm{ cm}^2)$",
                             ["SHM_eta0", "SHM_eta1"]),
                            ("LUX2016", ["LUX2016zero","LUX2016zero", "LUX2016five", "LUX2016many"]),
                            ("PandaX", ["PandaX"]), ("CDMSlite2016", ["CDMSlite2016"]),
                            ("CDMS_Snolab_GeHV", ["CDMS Snolab Ge HV"]),
                            ("LZ", ["LZ"]),("Darwin", ["Darwin"]),
                            ("CDMS_Snolab_SiHV", ["CDMS Snolab Si HV"]),
                            ("CDMS_Snolab_GeiZip", ["CDMS Snolab Ge iZip"]),
                            ("PICO_500", ["PICO(500L)"]),("DarkSideG2", ["DarkSide G2"]),
                            ("PICO_60", ["PICO(60L)"])
                            ])
""" Transparency parameter for filling regions, depending on the quenching factor.
"""
transp_list = [0.6, 0.4]
transparency = {0.4: transp_list[0], 0.09: transp_list[0],
                (0.4, 0.09): transp_list[0], (0.09, 0.4): transp_list[0],
                0.3: transp_list[1], 0.06: transp_list[1],
                (0.3, 0.06): transp_list[1], (0.06, 0.3): transp_list[1]}


def confidence_level(sigma):
    return erf(sigma/np.sqrt(2))


def sigma_dev(CL):
    return np.sqrt(2) * erfinv(CL)


def chi_squared(dof, CL=ConfidenceLevel):
    return chi2.ppf(CL, dof)


def chi_squared1(CL=ConfidenceLevel):
    # special case for dof = 1
    return sigma_dev(CL)**2


def import_file(full_path_to_module):
    """ Imports Python module from file.
    """
    import os
    import sys
    directory, module_name = os.path.split(full_path_to_module)
    module_name = os.path.splitext(module_name)[0]

    path = list(sys.path)
    sys.path.insert(0, directory)
    try:
        module = __import__(module_name)
        return module
    finally:
        sys.path[:] = path  # restore


def FileNameTail(fp, fn, mPhi):
    """ Gives a file name-tail that is added to each output file, to distinguish
    between different input parameters.
    """
    if mPhi == mPhiRef:
        mPhi_string = ""
    else:
        mPhi_string = "_mPhi" + str(math.trunc(mPhi))
    fnfp = fn/fp
    fnfp_string = "_fnfp"
    if fnfp == 1.:
        fnfp_string = "_fnfp1"
    elif abs(fnfp) < 1:
        if fnfp < 0:
            fnfp_string += "_neg0" + str(math.trunc(round(10 * abs(fnfp))))
        else:
            fnfp_string += "0" + str(math.trunc(round(10 * abs(fnfp))))
    else:
        if fnfp < 0:
            fnfp_string += "_neg" + str(math.trunc(round(abs(fnfp))))
        else:
            fnfp_string += str(math.trunc(round(abs(fnfp))))
    return mPhi_string + fnfp_string


def OutputDirectory(output_main_dir, scattering_type, mPhi, delta):
    """ Gives the name of the output directory for the given input parameters.
    Input:
        output_main_dir: string, optional
            Name of main output directory.
        scattering_type: string
            'SI' for spin-dependent, 'SDPS' for pseudo-scalar, 'SDAV' for axial-vector.
        mPhi: float
            Mass of mediator.
        delta: float
            DM mass split.
    Returns:
        out_dir: string
            Full path of output directory.
    """
    out_dir = output_main_dir
    if mPhi == mPhiRef:
        out_dir += "Contact"
    else:
        out_dir += "LongRange"
    out_dir += scattering_type + "_delta_"
    if delta < 0:
        out_dir += "neg"
    out_dir += str(math.trunc(abs(delta))) + "/"
    return out_dir


def Output_file_name(exper_name, scattering_type, mPhi, mx, fp, fn, delta, HALO_DEP,
                     filename_tail, OUTPUT_MAIN_DIR, quenching=None):
    """ Gives the name of the output file name for the given input parameters.
    Input:
        exper_name: string
            Name of experiment.
        scattering_type: string
            'SI' for spin-dependent, 'SDPS' for pseudo-scalar, 'SDAV' for axial-vector.
        mPhi: float
            Mass of mediator.
        mx: float
            DM mass.
        fp and fn: float
            Couplings to proton and neutron.
        delta: float
            DM mass split.
        confidence_levels: list
            List of confidence levels.
        HALO_DEP: bool
            Whether the analysis is halo-dependent or halo-independent.
        filename_tail: string
            Tag to be added to the file name.
        OUTPUT_MAIN_DIR: string
            Name of main output directory.
        quenching: float, optional
            quenching factor, needed for experiments that can have multiple options.
    """
    output_dir = OutputDirectory(OUTPUT_MAIN_DIR, scattering_type, mPhi, delta)
    output_file_no_extension = "./" + output_dir + "UpperLimit_" + exper_name

    if HALO_DEP:
        output_file_no_extension += "_mxsigma"
        if vesc != default_vesc:
            output_file_no_extension += "_vesc" \
                + str(math.trunc(round(vesc)))
        if vobs != default_vobs:
            output_file_no_extension += "_vobs" \
                + str(math.trunc(round(vobs)))
    else:
        output_file_no_extension += "_mx_" + str(mx) + "GeV"

    output_file_no_extension += FileNameTail(fp, fn, mPhi) + filename_tail

    if quenching is not None:
        output_file_no_extension += "_q" + str(quenching)
    print(output_file_no_extension)
    return output_file_no_extension


def MultiExper_Output_file_name(multiexper_input, scattering_type, mPhi, mx, fp, fn, delta,
                                filename_tail, OUTPUT_MAIN_DIR, quenching=None):
    """ Gives the name of the output file name for the given input parameters.
    Input:
        multiexper_input: string
            Names of experiments.
        scattering_type: string
            'SI' for spin-dependent, 'SDPS' for pseudo-scalar, 'SDAV' for axial-vector.
        mPhi: float
            Mass of mediator.
        mx: float
            DM mass.
        fp and fn: float
            Couplings to proton and neutron.
        delta: float
            DM mass split.
        confidence_levels: list
            List of confidence levels.

        filename_tail: string
            Tag to be added to the file name.
        OUTPUT_MAIN_DIR: string
            Name of main output directory.
        quenching: float, optional
            quenching factor, needed for experiments that can have multiple options.
    """
    output_dir = OutputDirectory(OUTPUT_MAIN_DIR, scattering_type, mPhi, delta)
    output_file_no_extension = "./" + output_dir + "MultiExperiment_EHI_"
    for x in range(0, multiexper_input.size):
        output_file_no_extension += multiexper_input[x]

    output_file_no_extension += "_mx_" + str(mx) + "GeV"

    output_file_no_extension += FileNameTail(fp, fn, mPhi) + filename_tail

    if quenching is not None:
        output_file_no_extension += "_q" + str(quenching)
    print(output_file_no_extension)
    return output_file_no_extension


def Plot_file_name(HALO_DEP, scattering_type, mPhi, fp, fn, delta,
                   filename_tail, OUTPUT_MAIN_DIR, mx=None):
    """ Gives the name of the plot file name for the given input parameters.
    Input:
        HALO_DEP: bool
            Whether the analysis is halo-dependent or halo-independent.
        scattering_type: string
            'SI' for spin-dependent, 'SDPS' for pseudo-scalar, 'SDAV' for axial-vector.
        mPhi: float
            Mass of mediator.
        fp and fn: float
            Couplings to proton and neutron.
        delta: float
            DM mass split.
        filename_tail: string
            Tag to be added to the file name.
        OUTPUT_MAIN_DIR: string
            Name of main output directory.
        mx: float, optional
            DM mass.
    """
    out_dir = OUTPUT_MAIN_DIR + "PLOTS_"
    if mPhi == mPhiRef:
        scattering = "Contact"
    else:
        scattering = "LongRange"
    scattering += scattering_type
    out_dir += scattering + "_Py/"

    if HALO_DEP:
        file_name = out_dir + "HaloDep_"
    else:
        file_name = out_dir + "HaloIndep_"

    file_name += scattering + "_delta_"
    if delta < 0:
        file_name += "neg"
    file_name += str(math.trunc(abs(delta))) + FileNameTail(fp, fn, mPhi) + \
        filename_tail + ".pdf"
    return file_name


def Gaussian(x, mu, sigma):
    """ Gaussian resolution function.
    """
    return np.exp(-(x-mu)**2 / (2 * sigma**2)) / (np.sqrt(2 * pi) * sigma)


def GPoisson(x, nu, sigma):
    """ Resolution function for the Xe experiment; it is a combination of
    Poisson and Gaussian resolution.
        nu is the expected # of events.
    """
    eps = 1.e-4
    n = 1
    add = nu * np.exp(-(x-1.)**2 / (2 * sigma**2))
    summation = 0.
    nfact = 1  # factorial
    try:
        while add > eps * (summation+add):
            #print("TEST", add, summation, x, nu, sigma)
            summation += add
            n += 1
            nfact *= n
            add = 1. * nu**n / nfact * np.exp(-(x-n)**2 / (2. * n * sigma**2)) / np.sqrt(n)
    except TypeError:
        return 0.
    result = summation * np.exp(-nu) / np.sqrt(2 * np.pi) / sigma
    #print("GPoisson: ", result)
    return result


def HelmFF(ER, A, mT):
    """ Helm Form Factor. See http://arxiv.org/abs/hep-ph/0608035.
    Input:
        ER: float
            Recoil energy.
        A: int
            Target mass number.
        mT: float
            Target nuclide mass.
    """
    q = np.sqrt(2e-6 * mT * ER)
    s = 0.9
    ha = 0.52
    hc = 1.23 * A**(1./3) - 0.6  # half-density radius
    cpa = 7./3. * (pi * ha)**2
    rsq = hc**2 + cpa
    r1tmp = rsq - 5. * s**2
    r1 = list(map(lambda r, s: np.sqrt(r) if r > 0 else np.sqrt(s), r1tmp, rsq))
    x = np.abs(q * r1 * fermiGeV)
    y = q * s * fermiGeV
    f = np.array(list(map(lambda i: 3.0 * (np.sin(i) - i * np.cos(i))/i**3 if i > 5.e-8
                 else 1. - i**2 * (-0.1 + i**2 *
                                   (1./200. + i**2 * (-1./15120. + i**2/1330560.))), x)))
    return f**2 * np.exp(-y**2)


def GaussianFFSD(ER, A, mT):
    """ Gaussian Form Factor for spin-dependent interactions.
    Input:
        ER: float
            Recoil energy.
        A: int
            Target mass number.
        mT: float
            Target nuclide mass.
    """
    q = np.sqrt(2e-6 * mT * ER)
    R = 0.92 * A**(1./3) + 2.68 - 0.78 * np.sqrt((A**(1./3) - 3.8)**2 + 0.2)
    x = np.abs(q * R * fermiGeV)
    return np.exp(-x**2 / 4.)

""" Form factor options are used to select the correct FF depending on the type of
interaction (spin-independent, spin-dependent with axial-vector or pseudo-scalar
coupling).
"""
FFSI_options = {'HelmFF': HelmFF,
                }
FFSD_options = {'GaussianFFSD': GaussianFFSD,
                }
FF_options = {'SI': FFSI_options,
              'SDPS': FFSD_options,
              'SDAV': FFSD_options,
              }


def ERecoil_ratio(mT_1, mT_2, mx, quenching_1, quenching_2):
    mu_1 = mT_1 * mx / (mT_1 + mx)
    mu_2 = mT_2 * mx / (mT_2 + mx)
    return mu_2**2 / mu_1**2 * mT_1 / mT_2 * quenching_2 / quenching_1


def VMin(ER, mT, mx, delta):
    """ Minimum velocity for a given recoil energy.
    Input:
        ER: float
            Recoil energy.
        mT: float
            Target nuclide mass.
        mx: float
            DM mass.
        delta: float
            DM mass split.
    Returns:
        vmin: float
    """
    muT = mx * mT / (mx + mT)
    return SpeedOfLight * 1.e-3 / np.sqrt(2. * ER * mT) * abs(delta + ER * mT / muT)


def VminDelta(mT, mx, delta):
    muT = mx * mT / (mx + mT)
    return SpeedOfLight / 500. * np.sqrt(delta / 2. / muT) if delta > 0 \
        else np.array([0] * len(mT))


def ERecoilBranch(vmin, mT, mx, delta, sign):
    """ Recoil energy for given vmin.
    Input:
        vmin: float
            Minimum DM velocity vmin.
            Target nuclide mass.
        mT: float
            Target nuclide mass.
        mx: float
            DM mass.
        delta: float
            DM mass split.
        sign: +1 or -1
            Corresponds to the upper and lower branch, respectively.
    Returns:
        ER: float
    """
    muT = mx * mT / (mx + mT)
    if delta > 0:
        vdel = np.sqrt(2. * delta / muT)
    else:
        vdel = np.array([0.])
    if vmin >= min(vdel):
        return 1.e6 * muT**2 / (2.*mT) * \
            (vmin / SpeedOfLight +
             sign * np.sqrt((vmin / SpeedOfLight)**2 - 2.*delta / muT * 1.e-6))**2
    else:
        return 1.e6 * muT**2 / (2.*mT) * \
            (vdel / SpeedOfLight +
             sign * np.sqrt((vdel / SpeedOfLight)**2 - 2.*delta / muT * 1.e-6))**2

def dERecoildVmin(vmin, mT, mx, delta, sign):
    """ Derivative of recoil energy ER with respect to velocity vmin
    Input:
        Input:
        vmin: float
            Minimum DM velocity vmin.
            Target nuclide mass.
        mT: float
            Target nuclide mass.
        mx: float
            DM mass.
        delta: float
            DM mass split.
        sign: +1 or -1
            Corresponds to the upper and lower branch, respectively.
    Returns:
        d ER/d vmin: float
    """
    muT = mx * mT / (mx + mT)
    if delta > 0:
        vdel = np.sqrt(2. * delta / muT)
    else:
        vdel = np.array([0.])

    if vmin != 0 and vmin >= min(vdel):
        sqrt_factor = np.sqrt(1. - 2.*delta / (muT * vmin**2) * SpeedOfLight**2 * 1.e-6)
        ret = sign * muT**2 * vmin / mT * (1. + sign * sqrt_factor)**2 / sqrt_factor
    elif vmin == 0 and delta == 0:
        ret = muT**2 * vmin / mT
    elif vmin == 0 and delta < 0:
        ret = sign * (- delta * muT) / (np.sqrt(- 2.0 * delta * SpeedOfLight**2 * 1.e-6 / muT) * mT) 
    else:
        ret = 0.
    return ret


def eta0Maxwellian(vmin, vobs, v0bar, vesc):
    """ Velocity integral eta0 in a Standard Halo Model with Maxwellian velocity
    distribution.
    Input:
        vmin: float
            Minumum DM velocity.
        vobs: float
            Observed velocity.
        v0bar: float
            Velocity dispersion.
        vesc: float
            Excape velocity.
    Returns:
        eta0: float
    """
    x = vmin/v0bar
    y = vobs/v0bar
    z = vesc/v0bar
    erfz = erf(z)
    sqrt_pi = np.sqrt(pi)
    exp_z_sq = np.exp(-z**2)
    exp_z_sq_z = np.exp(-z**2) * z
    eta = list(map(lambda i: -2. * exp_z_sq / sqrt_pi - erf(i-y) / (2.*y) +
                   erf(i+y) / (2.*y)
                   if i + y <= z
                   else exp_z_sq * (i - y - z) / (sqrt_pi * y) - erf(i-y) / (2.*y) +
                   erfz / (2.*y)
                   if i - y <= z < i + y
                   else 0, x))
    return eta / (-2. * exp_z_sq_z / sqrt_pi + erfz) / v0bar


def eta1Maxwellian(vmin, vobs, v0bar, vesc):
    """ Same as eta0Maxwellian, but this is the modulation velocity integral
    eta1 = d eta0 / d vobs * delta_v.
        delta_v = v_Earth * cos(gamma) where the velocity of the Earth is
    v_Earth = 30 km/s and is inclined at an angle of gamma = 60 deg wrt the
    galactic plane.
    Returns:
        eta1: float
    """
    x = vmin/v0bar
    y = vobs/v0bar
    z = vesc/v0bar
    delta_v = 15.  # 30 * cos(60 * pi / 180)
    erfz = erf(z)
    sqrt_pi = np.sqrt(pi)
    exp_z_sq = np.exp(-z**2)
    exp_z_sq_z = np.exp(-z**2) * z
    erf_z = erf(z)
    eta = list(map(lambda i: (np.exp(-(i+y)**2) + np.exp(-(i-y)**2)) / (sqrt_pi * y) +
                   (erf(i-y) - erf(i+y)) / (2 * y**2)
                   if i + y <= z
                   else exp_z_sq * (-i + z) / (sqrt_pi * y**2) +
                   np.exp(-(i-y)**2) / (sqrt_pi * y) + (erf(i-y) - erf_z) / (2 * y**2)
                   if i - y <= z < i + y
                   else 0, x))
    return eta / (-2. * exp_z_sq_z / sqrt_pi + erfz) * delta_v / v0bar**2


def MaximumGapC0scaled(x, mu_over_x):
    """ Scaled probability C0 of the maximum gap size being smaller than a particular
    value of x. Based on Maximum Gap Method (Eq 2 of PHYSICAL REVIEW D 66, 032005, 2002).
    Input:
        x: float
            The size of the maximum gap in the random experiment.
        mu_over_x: float
            mu/x, where mu is the total expected number of events.
    """
    if mu_over_x < 1.:
        return 1.
    elif 1. <= mu_over_x < 2.:
        return 1. - np.exp(-x) * (1. + mu_over_x * x - x)
    else:
        l = np.array([((k - mu_over_x) * x)**(k-1) * np.exp(-k * x) / math.factorial(k) *
                     (x * (mu_over_x - k) + k)
                     for k in range(1, math.trunc(np.floor(mu_over_x)))])
        return 1. - l.sum()


def Rebin_data(data, error):
    """ Rebins the data from multiple bins into a single bin.
    Input:
        data, error: ndarray
            Lists containing the data and error from each bin.
    Returns:
        data_rebinned, error_rebinned: float
    """
    data_rebinned = sum(data)
    error_rebinned = np.sqrt(sum(error**2))
    return data_rebinned, error_rebinned


if __name__ == "__main__":
    vmin = np.linspace(1, 1000, 1000)
    mx = 1.3
    delta = -200
    mT = np.array([65.134, 66.995, 67.9278, 68.8571, 70.7203])
    ER_plus = np.array([ERecoilBranch(vmin, mT_i, mx, delta, 1) for mT_i in mT])
    ER_minus = np.array([ERecoilBranch(vmin, mT_i, mx, delta, -1) for mT_i in mT])
    import matplotlib.pyplot as plt
    plt.close()
    for ER in ER_plus:
        plt.plot(vmin, ER)
    for ER in ER_minus:
        plt.plot(vmin, ER)
    plt.plot(vmin, 2.7 * np.ones_like(vmin))
    plt.plot(vmin, 3 * np.ones_like(vmin))
    ER = np.linspace(2, 6, 100)
    plt.plot(490 * np.ones_like(ER), ER)
    plt.plot(495 * np.ones_like(ER), ER)
    plt.show()
