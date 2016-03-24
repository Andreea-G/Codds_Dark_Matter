"""
Copyright (c) 2015 Andreea Georgescu

Created on Thu Nov 20 22:52:11 2014

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

from globalfnc import *
from formfactors_eft import *
import numpy as np
from numpy import pi
from math import factorial, log10
from scipy.special import gamma, gammaincc, gammainc
from scipy import integrate, interpolate
from scipy.optimize import fsolve, minimize
# from scipy.special import lambertw
from lambertw import *
from math import log
import parallel_map as par
import matplotlib.pyplot as plt

INPUT_DIR = "Data/"


class Experiment:
    """ Base class that implements the core of the algorithm, common to all experiments
    and data analyses.
    Input:
        exper_name: string
            Name of experiment.
        scattering_type: string
            The type of scattering. Can be
                - 'SI' (spin-independent)
                - 'SDAV' (spin-independent, axial-vector)
                - 'SDPS' (spin-independent, pseudo-scalar)
        mPhi: float, optional
            Mass of the mediator. If not given, it corresponds to contact interaction.
    """
    def __init__(self, exper_name, scattering_type, mPhi=mPhiRef):
        # Import the data from Python module
        module = import_file(INPUT_DIR + exper_name + ".py")
        self.name = exper_name
        self.scattering_type = scattering_type
        self.energy_resolution_type = module.energy_resolution_type
        self.EnergyResolution = module.EnergyResolution
        self.ResolutionFunction = Gaussian if \
            self.energy_resolution_type == "Gaussian" else GPoisson

        self.mPhi = mPhi
        # angular momenta of target nuclides
        self.J = module.target_nuclide_JSpSn_list[:, 0]
        self.numT = module.num_target_nuclides
        if "SD" in self.scattering_type:
            # masses of target nuclides
            self.mT = np.array([module.target_nuclide_mass_list[i]
                                for i in range(self.numT) if self.J[i] != 0])
            # atomic mass numbers of target nuclides
            self.A = np.array([module.target_nuclide_AZC_list[i, 0]
                               for i in range(self.numT) if self.J[i] != 0])
            # atomic numbers of target nuclides
            self.Z = np.array([module.target_nuclide_AZC_list[i, 1]
                               for i in range(self.numT) if self.J[i] != 0])
            self.mass_fraction = np.array([module.target_nuclide_AZC_list[i, 2]
                                           for i in range(self.numT) if self.J[i] != 0])
            # Sp * sqrt((2*J + 1) * (J + 1) / (4*pi*J)) where Sp is the z-projection
            # of proton spin. Same for Sn.
            self.SpScaled = np.array([module.target_nuclide_JSpSn_list[i, 1]
                                      for i in range(self.numT) if self.J[i] != 0])
            self.SnScaled = np.array([module.target_nuclide_JSpSn_list[i, 2]
                                      for i in range(self.numT) if self.J[i] != 0])
            self.J = np.array([j for j in self.J if j != 0])
            self.numT = np.size(self.J)
        else:
            self.mT = module.target_nuclide_mass_list
            self.A = module.target_nuclide_AZC_list[:, 0]
            self.Z = module.target_nuclide_AZC_list[:, 1]
            self.mass_fraction = module.target_nuclide_AZC_list[:, 2]
            self.SpScaled = module.target_nuclide_JSpSn_list[:, 1]
            self.SnScaled = module.target_nuclide_JSpSn_list[:, 2]

        # modulated or unmodulated velocity integrals
        # for a Maxwellian velocity distribution
        if module.modulated is False:
            self.etaMaxwellian = eta0Maxwellian
        else:
            self.etaMaxwellian = eta1Maxwellian

        # Form factors
        FF_default = np.array([[lambda y: 0]*2]*2)
        self._FFSigmaPPJ_function_list = \
            np.array([FFSigmaPPJ.get((np.trunc(A), np.trunc(Z)), FF_default)
                      for A, Z in zip(self.A, self.Z)])
        self._FFSigmaPJ_function_list = \
            np.array([FFSigmaPJ.get((np.trunc(A), np.trunc(Z)), FF_default)
                      for A, Z in zip(self.A, self.Z)])
        # form factor
        self.FF = FF_options[self.scattering_type][module.FF[scattering_type]]
        # tests if it's been implemented in Fitzpatrick et al paper
        self.ffelemQ = FFElementQ(self.Z)
        if (self.ffelemQ == 1).all():
            print("Effective Field Theory")
            self._CrossSectionFactors_SDAV = self._CrossSectionFactors_SDAV_EFT
            self._CrossSectionFactors_SDPS = self._CrossSectionFactors_SDPS_EFT
        else:
            print("mixed")
            self._CrossSectionFactors_SDAV = self._CrossSectionFactors_SDAV_Mixed
            self._CrossSectionFactors_SDPS = self._CrossSectionFactors_SDPS_Mixed

        CrossSectionFactors_options = {'SI': self._CrossSectionFactors_SI,
                                       'SDPS': self._CrossSectionFactors_SDPS,
                                       'SDAV': self._CrossSectionFactors_SDAV,
                                       }
        self.CrossSectionFactors = CrossSectionFactors_options[self.scattering_type]
        if self.energy_resolution_type == "Dirac":
            self.IntegratedResponseSHM = self._IntegratedResponseSHM_Dirac
        else:
            self.IntegratedResponseSHM = self._IntegratedResponseSHM_Finite

        self.QuenchingFactor = module.QuenchingFactor
        self.Efficiency = module.Efficiency
        self.Efficiency_ER = module.Efficiency_ER
        self.Ethreshold = module.Ethreshold
        self.Emaximum = module.Emaximum
        self.ERmaximum = module.ERmaximum
        self.Exposure = module.Exposure

        self.count_diffresponse_calls = 0
        self.count_response_calls = 0

        self._bsq = 41.467/(45. * self.A**(-1./3) - 25. * self.A**(-2./3)) * fermiGeV**2
        self._y_over_ER = 2.e-6 * self.mT * self._bsq / 4.
        self._cross_sec_factors_SDPS = (SpeedOfLight/v0bar)**4 * \
            self.mass_fraction * self.mT**2

    def _FormFactor(self, ER):
        """ Form factor including spin-dependence for the spin-dependent interaction, when
        the EFT FF is not used.
        Input:
            ER: float or ndarray
                Recoil energy.
        Returns:
            result: ndarray
        """
        result = (4./3. * (4. * pi)/(2 * self.J + 1.)) * self.FF(ER, self.A, self.mT)
        return result

    def _CrossSectionFactors_SI(self, ER, mx, fp, fn, delta):
        """ Factors specific to the cross-section for the spin-independent interaction.
            Correspond to 1/sigma_p * 1/mT * v**2 * d sigma / d ER.
        """
        mu_p = ProtonMass * mx / (ProtonMass + mx)
        return self.mass_fraction * 1./(2.*mu_p**2) * \
            mPhiRef**4 / (4. * self.mT**2 * (ER + self.mPhi**2/(2. * self.mT))**2) * \
            ((self.Z + (self.A - self.Z) * fn/fp)**2) * self.FF(ER, self.A, self.mT)

    def _FFPSnormlalized(self, ER, N1, N2):
        """ Form factors for the PS interaction.
        See Eq. 2.19 of arXiv:1502.07682 and Appendix A.3. of arXiv:1203.3542.
        Input:
            ER: float or ndarray
                Recoil energy.
            N1, N2: 0 or 1 for proton or neutron respectively.
        """
        y = ER * self._y_over_ER
        l = np.empty(self.numT)
        for i in range(self.numT):
            l[i] = self._FFSigmaPPJ_function_list[i, N1, N2](y[i])
        return l * np.exp(-2. * y)

    def _FFAVnormlalized(self, ER, N1, N2):
        """ Form factors for the AV interaction.
        See Eq. 2.18 of arXiv:1502.07682 and Appendix A.3. of arXiv:1203.3542.
        Input:
            ER: float or ndarray
                Recoil energy.
            N1, N2: 0 or 1 for proton or neutron respectively.
        """
        y = ER * self._y_over_ER
        l = np.empty(self.numT)
        for i in range(self.numT):
            l[i] = 1./3 * (self._FFSigmaPPJ_function_list[i, N1, N2](y[i]) +
                           self._FFSigmaPJ_function_list[i, N1, N2](y[i]))
        return l * np.exp(-2. * y)

    def _CrossSectionFactors_SDPS_EFT(self, ER, mx, fp, fn, delta):
        """ Factors specific to the cross-section for the spin-dependent pseudo-scalar
        interaction, with EFT form factors.
            Correspond to 1/sigma_p * 1/mT * v**2 * d sigma / d ER.
        """
        mu_p = ProtonMass * mx / (ProtonMass + mx)
        return 1.e-12 * ER**2 * 3./(8. * mu_p**6) * \
            mPhiRef**4 / (4. * self.mT**2 * (ER + self.mPhi**2 / 2 / self.mT)**2) * \
            self._cross_sec_factors_SDPS * \
            (self._FFPSnormlalized(ER, 0, 0) +
                2 * fn/fp * self._FFPSnormlalized(ER, 0, 1) +
                (fn/fp)**2 * self._FFPSnormlalized(ER, 1, 1))

    def _CrossSectionFactors_SDPS_Mixed(self, ER, mx, fp, fn, delta):
        """ Factors specific to the cross-section for the spin-dependent pseudo-scalar
        interaction, with mixed form factors (some target elements from EFT paper,
        and some not).
            Correspond to 1/sigma_p * 1/mT * v**2 * d sigma / d ER.
        """
        mu_p = ProtonMass * mx / (ProtonMass + mx)
        return 1.e-12 * ER**2 * 3./(8. * mu_p**6) * \
            mPhiRef**4 / (4. * self.mT**2 * (ER + self.mPhi**2 / 2 / self.mT)**2) * \
            self._cross_sec_factors_SDPS * \
            (self.ffelemQ * (self._FFPSnormlalized(ER, 0, 0) +
                             2 * fn/fp * self._FFPSnormlalized(ER, 0, 1) +
                             (fn/fp)**2 * self._FFPSnormlalized(ER, 1, 1)) +
             (1 - self.ffelemQ) * (self.SpScaled + self.SnScaled * fn/fp)**2 *
             self._FormFactor(ER))

    def _CrossSectionFactors_SDAV_EFT(self, ER, mx, fp, fn, delta):
        """ Factors specific to the cross-section for the spin-dependent axial-vector
        interaction, with EFT form factors.
            Correspond to 1/sigma_p * 1/mT * v**2 * d sigma / d ER.
        """
        mu_p = ProtonMass * mx / (ProtonMass + mx)
        return self.mass_fraction / (2 * mu_p**2) * \
            mPhiRef**4 / (4. * self.mT**2 * (ER + self.mPhi**2/(2 * self.mT))**2) * \
            (self._FFAVnormlalized(ER, 0, 0) +
             2 * fn/fp * self._FFAVnormlalized(ER, 0, 1) +
             (fn/fp)**2 * self._FFAVnormlalized(ER, 1, 1))

    def _CrossSectionFactors_SDAV_Mixed(self, ER, mx, fp, fn, delta):
        """ Factors specific to the cross-section for the spin-dependent axial-vector
        interaction, with mixed form factors (some target elements from EFT paper,
        and some not).
            Correspond to 1/sigma_p * 1/mT * v**2 * d sigma / d ER.
        """
        mu_p = ProtonMass * mx / (ProtonMass + mx)
        return self.mass_fraction / (2 * mu_p**2) * \
            mPhiRef**4 / (4. * self.mT**2 * (ER + self.mPhi**2/(2 * self.mT))**2) * \
            (self.ffelemQ * (self._FFAVnormlalized(ER, 0, 0) +
                             2 * fn/fp * self._FFAVnormlalized(ER, 0, 1) +
                             (fn/fp)**2 * self._FFAVnormlalized(ER, 1, 1)) +
             (1-self.ffelemQ) * (self.SpScaled + self.SnScaled * fn/fp)**2 *
             self._FormFactor(ER))

    def Resolution(self, Eee, qER):
        """ Energy resolution G(Eee, ER).
        Input:
            Eee: float or ndarray
                Measured energy (electron equivalent).
            qER: float or ndarray
                q * ER for quenching factor q and recoil energy ER.
        """
        return self.ResolutionFunction(Eee, qER, self.EnergyResolution(qER))

    def DifferentialResponseSHM(self, Eee, qER, const_factor):
        """ Differential response function d**2 R / (d Eee d ER)
        Input:
            Eee: float or ndarray
                Measured energy (electron equivalent).
            qER: float or ndarray
                q * ER for quenching factor q and recoil energy ER.
            const_factor: ndarray
                Factors entering the differential response that do not depend on Eee.
        """
        self.count_diffresponse_calls += 1
        r_list = const_factor * self.Efficiency(Eee) * \
            self.ResolutionFunction(Eee, qER, self.EnergyResolution(qER))
        return r_list.sum()

    def _ResponseSHM_Finite(self, ER, Eee1, Eee2, mx, fp, fn, delta):
        """ Response function: integral over d**2 R / (d Eee d ER) between
        measured energies Eee1 and Eee2, as a function of recoil energy ER.
            For any finite resolution function (i.e. other than Dirac Delta).
        """
        self.count_response_calls += 1
        q = self.QuenchingFactor(ER)
        qER = q * ER
        vmin = VMin(ER, self.mT, mx, delta)
        const_factor = 1.e-6 * \
            kilogram * self.CrossSectionFactors(ER, mx, fp, fn, delta) * \
            self.Efficiency_ER(ER) * self.etaMaxwellian(vmin, vobs, v0bar, vesc)
        result = integrate.quad(self.DifferentialResponseSHM, Eee1, Eee2,
                                args=(qER, const_factor), epsrel=PRECISSION, epsabs=0)[0]
        return result

    def _ResponseSHM_Dirac(self, ER, Eee1, Eee2, mx, fp, fn, delta):
        """ Response function: integral over d**2 R / (d Eee d ER) between
        measured energies Eee1 and Eee2, as a function of recoil energy ER.
            For Dirac Delta resolution function.
        """
        self.count_response_calls += 1
        q = self.QuenchingFactor(ER)
        qER = q * ER
        vmin = VMin(ER, self.mT, mx, delta)
        integrated_delta = 1. if Eee1 <= qER < Eee2 else 0.
        r_list = 1.e-6 * kilogram * \
            self.CrossSectionFactors(ER, mx, fp, fn, delta) * \
            self.Efficiency(Eee1, qER) * self.Efficiency_ER(qER) * \
            integrated_delta * self.etaMaxwellian(vmin, vobs, v0bar, vesc)
        r_list_sum = r_list.sum()
        return r_list_sum

    def _IntegratedResponseSHM_Finite(self, Eee1, Eee2, mx, fp, fn, delta):
        """ Integrated Response Function between measured energies Eee1 and Eee2,
        and all of recoil energies ER.
            For any finite resolution function (i.e. other than Dirac Delta).
        """
        vmax = vesc + vobs
        vdelta = VminDelta(self.mT, mx, delta)
        ER_plus_list = [ERecoilBranch(vmax, mT, mx, delta, 1) if vmax > vd else 0
                        for mT, vd in zip(self.mT, vdelta)]
        ER_minus_list = [ERecoilBranch(vmax, mT, mx, delta, -1) if vmax > vd else 1.e6
                         for mT, vd in zip(self.mT, vdelta)]

        ER_plus = min(np.max(ER_plus_list), self.ERmaximum)
        ER_minus = np.min(ER_minus_list)
        midpoints = []
        if ER_minus < Eee1 < ER_plus:
            midpoints += [Eee1]
        if ER_minus < Eee2 < ER_plus:
            midpoints += [Eee2]
        if ER_minus < ER_plus:
            integr = integrate.quad(self._ResponseSHM_Finite, ER_minus, ER_plus,
                                    args=(Eee1, Eee2, mx, fp, fn, delta),
                                    points=midpoints, epsrel=PRECISSION, epsabs=0)
#            integr = integrate.dblquad(self.DifferentialResponseSHM, ER_minus, ER_plus, \
#                                       lambda Eee: Eee1, lambda Eee: Eee2, \
#                                       args=(mx, fp, fn, delta), epsrel = PRECISSION,
#                                       epsabs = 0)
            return integr[0]
        else:
            return 0.

    def _IntegratedResponseSHM_Dirac(self, Eee1, Eee2, mx, fp, fn, delta):
        """ Integrated Response Function between measured energies Eee1 and Eee2,
        and all recoil energies ER.
            For Dirac Delta resolution function.
        """
        vmax = vesc + vobs
        vdelta = VminDelta(self.mT, mx, delta)
        ER_plus_list = [ERecoilBranch(vmax, mT, mx, delta, 1) if vmax > vd else 0
                        for mT, vd in zip(self.mT, vdelta)]
        ER_minus_list = [ERecoilBranch(vmax, mT, mx, delta, -1) if vmax > vd else 1.e6
                         for mT, vd in zip(self.mT, vdelta)]

        # TODO! This is only valid for quenching factor 1!!! Extend to arbitrary q!
        ER_plus = min(min(np.max(ER_plus_list), self.ERmaximum), Eee2)
        ER_minus = max(np.min(ER_minus_list), Eee1)
        if ER_minus < ER_plus:
            integr = integrate.quad(self._ResponseSHM_Dirac, ER_minus, ER_plus,
                                    args=(Eee1, Eee2, mx, fp, fn, delta))
            return integr[0]
        else:
            return 0.


class PoissonExperiment(Experiment):
    """ Class for experiments with Poisson analysis.
    Input:
        exper_name: string
            Name of experiment.
        scattering_type: string
            Type of scattering.
        mPhi: float, optional
            Mass of the mediator.
        quenching_factor: float, optional
            Quenching factor. If not given, the default used is specified in the data
            modules.
    """
    def __init__(self, exper_name, scattering_type, mPhi=mPhiRef, quenching_factor=None):
        super().__init__(exper_name, scattering_type, mPhi)
        module = import_file(INPUT_DIR + exper_name + ".py")
        self.Expected_limit = module.Expected_limit

    def _PoissonUpperBoundSHM(self, mx, fp, fn, delta):
        vmax = vobs + vesc
        muT = self.mT * mx / (self.mT + mx)
        vdelta = SpeedOfLight / 500. * np.sqrt(delta / 2. / muT) if delta > 0 \
            else np.array([0] * self.numT)
        Eee_max_list = [ERecoilBranch(vmax, mT, mx, delta, 1) if vmax > vd else 0
                        for mT, vd in zip(self.mT, vdelta)]
        Eee_min_list = [ERecoilBranch(vmax, mT, mx, delta, -1) if vmax > vd else 1.e6
                        for mT, vd in zip(self.mT, vdelta)]

        Eee_max = np.max(Eee_max_list)
        Eee_min = max(self.Ethreshold, np.min(Eee_min_list))
        int_response = self.IntegratedResponseSHM(Eee_min, Eee_max, mx, fp, fn, delta)
        if int_response > 0:
            result = self.Expected_limit / self.Exposure / int_response
        else:
            result = np.inf
        print("mx = ", mx)
        print("result = ", result)
        return result

    def UpperLimit(self, fp, fn, delta, mx_min, mx_max, num_steps, output_file,
                   processes=None):
        """ Computes the upper limit in cross-section as a function of DM mass mx.
        Input:
            fp, fn: float
                Couplings to proton and neutron.
            delta: float
                DM mass split.
            mx_min, mx_max: float
                Minimum and maximum DM mass.
            num_steps: int
                Number of steps in the range of DM mass values.
            output_file: string
                Name of output file where the result will be printed.
        Returns:
            upperlimit: ndarray
                Table with the first row giving the base-10 log of the DM mass, and
                the second row giving the base-10 log of the upper limit in cross-section.
        """
        mx_list = np.logspace(np.log10(mx_min), np.log10(mx_max), num_steps)
        kwargs = ({'mx': mx,
                   'fp': fp,
                   'fn': fn,
                   'delta': delta}
                  for mx in mx_list)
        upper_limit = np.array(par.parmap(self._PoissonUpperBoundSHM, kwargs, processes))
        print("mx_list = ", mx_list)
        print("upper_limit = ", upper_limit)
        upper_limit = 1/conversion_factor * mx_list * upper_limit
        result = np.log10(np.transpose([mx_list, upper_limit]))
        result = result[result[:, 1] != np.inf]
        with open(output_file, 'ab') as f_handle:
            np.savetxt(f_handle, result)
        return result


class PoissonLikelihood(Experiment):
    """ Class for experiments with Poisson analysis.
    Input:
        exper_name: string
            Name of experiment.
        scattering_type: string
            Type of scattering.
        mPhi: float, optional
            Mass of the mediator.
        quenching_factor: float, optional
            Quenching factor. If not given, the default used is specified in the data
            modules.
    """
    def __init__(self, exper_name, scattering_type, mPhi=mPhiRef, quenching_factor=None):
        super().__init__(exper_name, scattering_type, mPhi)
        module = import_file(INPUT_DIR + exper_name + ".py")
        self.BinEdges_left = module.BinEdges_left
        self.BinEdges_right = module.BinEdges_right
        self.BinData = module.BinData
        self.BinSize = module.BinSize
        self.BinBkgr = module.BinBkgr
        self.chiSquared = chi_squared(1)
        self.BinExposure = module.BinExposure
    
    def _MinusLogLikelihood(self, mx, fp, fn, delta):
        """

        """
        rate_partials = [None] * (self.BinEdges_left.size)


        resp_integr = self.BinExposure[0] * conversion_factor / mx * self.IntegratedResponseSHM(
                            self.BinEdges_left[0], self.BinEdges_right[0], mx, fp, fn, delta)
        rate_partials[0] = resp_integr

        result = 0
        
        likemin = minimize(lambda y: 2.0 * (10.0 ** (-y) * rate_partials[0] +
                        self.BinBkgr[0] + log(factorial(self.BinData[0])) - self.BinData[0] *
                        log(10.0 ** (-y) * rate_partials[0] + self.BinBkgr[0])), 40., 
                        method = 'SLSQP', bounds = [(20., 100.)])
        
# Likelihood Analysis for SuperCDMS less T5
        result = fsolve(lambda y: 2.0 * (10.0 ** (-y) * rate_partials[0] +
                        self.BinBkgr[0] + log(factorial(self.BinData[0])) - self.BinData[0] *
                        log(10.0 ** (-y) * rate_partials[0] + self.BinBkgr[0]))
                        - self.chiSquared - likemin.fun, 36.0)
        
        print(result)
        return result

    def UpperLimit(self, fp, fn, delta, mx_min, mx_max, num_steps, output_file,
                   processes=None):
        """ Computes the upper limit in cross-section as a function of DM mass mx.
        Input:
            fp, fn: float
                Couplings to proton and neutron.
            delta: float
                DM mass split.
            mx_min, mx_max: float
                Minimum and maximum DM mass.
            num_steps: int
                Number of steps in the range of DM mass values.
            output_file: string
                Name of output file where the result will be printed.
        Returns:
            upperlimit: ndarray
                Table with the first row giving the base-10 log of the DM mass, and
                the second row giving the base-10 log of the upper limit in cross-section.
        """
        mx_list = np.logspace(np.log10(mx_min), np.log10(mx_max), num_steps)
        kwargs = ({'mx': mx,
                   'fp': fp,
                   'fn': fn,
                   'delta': delta}
                  for mx in mx_list)

        upper_limit = np.array(par.parmap(self._MinusLogLikelihood, kwargs, processes)).flatten()
        print("mx_list = ", mx_list)
        print("upper_limit = ", upper_limit)
        result = np.log10(np.transpose([mx_list, 10.0**-upper_limit]))
        result = result[result[:, 1] != np.inf]
        with open(output_file, 'ab') as f_handle:
            np.savetxt(f_handle, result)
        return result


class GaussianExperiment(Experiment):
    """ Class for experiments with Gaussian analysis.
    Input:
        exper_name: string
            Name of experiment.
        scattering_type: string
            Type of scattering.
        mPhi: float, optional
            Mass of the mediator.
        quenching_factor: float, optional
            Quenching factor. If not given, the default used is specified in the data
            modules.
    """
    def __init__(self, exper_name, scattering_type, mPhi=mPhiRef, quenching_factor=None):
        super().__init__(exper_name, scattering_type, mPhi)
        module = import_file(INPUT_DIR + exper_name + ".py")
        self.BinEdges_left = module.BinEdges_left
        self.BinEdges_right = module.BinEdges_right
        self.BinData = module.BinData
        self.BinError = module.BinError
        self.BinSize = module.BinSize
        self.chiSquared = chi_squared(self.BinData.size)
        if quenching_factor is not None:
            self.QuenchingFactor = lambda e: quenching_factor

    def _GaussianUpperBoundSHM(self, mx, fp, fn, delta, output_file):
        predicted = conversion_factor / mx * \
            np.array([self.IntegratedResponseSHM(i, j, mx, fp, fn, delta)
                      for i, j in zip(self.BinEdges_left, self.BinEdges_right)])
        sum_pred_squared = 1./self.BinSize**2 * sum((predicted/self.BinError)**2)
        sum_pred_bindata = 2./self.BinSize * \
            sum(predicted * self.BinData / self.BinError**2)
        sum_bindata_squared = sum((self.BinData/self.BinError)**2) - self.chiSquared
        result = (sum_pred_bindata + np.sqrt(sum_pred_bindata**2 -
                  4 * sum_pred_squared * sum_bindata_squared)) / (2 * sum_pred_squared)
        print("mx = ", mx)
        print("result = ", result)
        to_print = np.log10(np.array([[mx, result]]))
        with open(output_file, 'ab') as f_handle:
            np.savetxt(f_handle, to_print)
        return result

    def UpperLimit(self, fp, fn, delta, mx_min, mx_max, num_steps, output_file,
                   processes=None):
        """ Computes the upper limit in cross-section as a function of DM mass mx.
        """
        mx_list = np.logspace(np.log10(mx_min), np.log10(mx_max), num_steps)
        kwargs = ({'mx': mx,
                   'fp': fp,
                   'fn': fn,
                   'delta': delta,
                   'output_file': output_file}
                  for mx in mx_list)
        upper_limit = np.array(par.parmap(self._GaussianUpperBoundSHM, kwargs, processes)
                               ).flatten()
        print("mx_list = ", mx_list)
        print("upper_limit = ", upper_limit)
        result = np.log10(np.transpose([mx_list, upper_limit]))
        return result[result[:, 1] != np.inf]


class MaxGapExperiment(Experiment):
    """ Class for experiments using the Maximum Gap Method.
    Input:
        exper_name: string
            Name of experiment.
        scattering_type: string
            Type of scattering.
        mPhi: float, optional
            Mass of the mediator.
        quenching_factor: float, optional
            Quenching factor. If not given, the default used is specified in the data
            modules.
    """
    def __init__(self, exper_name, scattering_type, mPhi=mPhiRef, quenching_factor=None):
        super().__init__(exper_name, scattering_type, mPhi)
        module = import_file(INPUT_DIR + exper_name + ".py")
        self.ERecoilList = module.ERecoilList
        self.ElistMaxGap = \
            np.append(np.insert(np.array(list(filter(lambda x:
                                                     self.Ethreshold < x < self.Emaximum,
                                self.ERecoilList))), 0, self.Ethreshold), self.Emaximum)

    def MaximumGapUpperBoundSHM(self, mx, fp, fn, delta, output_file):
        print("mx = ", mx)
        xtable = np.array([self.IntegratedResponseSHM(i, j, mx, fp, fn, delta)
                           for i, j in zip(self.ElistMaxGap[:-1], self.ElistMaxGap[1:])])

        mu_scaled = xtable.sum()

        x_scaled = np.max(xtable)
        if x_scaled == 0:
            mu_over_x = np.inf
            result = [np.inf]
        else:
            mu_over_x = mu_scaled / x_scaled
            y_guess = np.real(-lambertw(-0.1 / mu_over_x, -1))

            y = fsolve(lambda x:
                       MaximumGapC0scaled(x, mu_over_x) - ConfidenceLevel, y_guess)
            result = y / x_scaled / self.Exposure
        print("mx = ", mx, "   mu_over_x = ", mu_over_x)
        print("xtable = ", xtable)
        print("yguess =", y_guess)
        print("result = ", result[0])
        to_print = np.log10(np.array([[mx, result[0]]]))
        with open(output_file, 'ab') as f_handle:
            np.savetxt(f_handle, to_print)
        return result

    def UpperLimit(self, fp, fn, delta, mx_min, mx_max, num_steps, output_file,
                   processes=None):
        """ Computes the upper limit in cross-section as a function of DM mass mx.
        """
        mx_list = np.logspace(np.log10(mx_min), np.log10(mx_max), num_steps)
#        mx_list= np.array([2.891, 3.029, 3.175, 3.327, 3.486, 3.654, 3.829, 4.012, 4.205, 4.406, \
#            4.617, 4.839, 5.071, 5.314, 5.569, 5.836, 6.116, 6.409, 6.716, 7.038, \
#            7.376, 7.729, 8.1, 8.488, 8.895, 9.322, 9.769, 10.237, 10.728, \
#                11.242, 11.781, 12.346, 12.938, 13.558, 14.208, 14.89, 15.603, \
#                16.352, 17.136, 17.957, 18.818, 19.72, 20.666, 21.657, 22.695, \
#                23.783, 24.924, 26.119, 27.371, 28.683, 30.058, 31.5, 33.01, 34.593, \
#                36.251, 37.989, 39.811])
        kwargs = ({'mx': mx,
                   'fp': fp,
                   'fn': fn,
                   'delta': delta,
                   'output_file': output_file}
                  for mx in mx_list)
        upper_limit = np.array(par.parmap(self.MaximumGapUpperBoundSHM, kwargs, processes)
                               ).flatten()
        upper_limit = 1./conversion_factor * mx_list * upper_limit
        print("mx_list = ", mx_list)
        print("upper_limit = ", upper_limit)
        result = np.log10(np.transpose([mx_list, upper_limit]))
        return result[result[:, 1] != np.inf]


class DAMAExperiment(Experiment):
    """ Class for finding the best-fit regions for the DAMA experiment.
    Input:
        exper_name: string
            Name of experiment.
        scattering_type: string
            Type of scattering.
        mPhi: float, optional
            Mass of the mediator.
        quenching_factor: float, optional
            Quenching factor. If not given, the default used is specified in the data
            modules.
    """
    def __init__(self, exper_name, scattering_type, mPhi=mPhiRef, quenching_factor=None):
        super().__init__(exper_name, scattering_type, mPhi)
        module = import_file(INPUT_DIR + exper_name + ".py")
        self.BinEdges = module.BinEdges
        self.BinData = module.BinData
        self.BinError = module.BinError
#        self._Rebin()
        self.BinEdges_left = self.BinEdges[:-1]
        self.BinEdges_right = self.BinEdges[1:]
        self.BinData = self.Exposure * self.BinData
        self.BinError = self.Exposure * self.BinError
        if quenching_factor is not None:
            self.QuenchingFactor = lambda e: quenching_factor

    def _Rebin(self, index=9):
        self.BinEdges = np.append(self.BinEdges[:index + 1],  self.BinEdges[-1])
        data, error = Rebin_data(self.BinData[index:], self.BinError[index:])
        self.BinData = np.append(self.BinData[:index], data)
        self.BinError = np.append(self.BinError[:index], error)
        print('BinEdges =', self.BinEdges)
        print('BinData =', self.BinData)
        print('BinError =', self.BinError)

    def _Predicted(self, mx, fp, fn, delta):
        return self.Exposure * conversion_factor / mx * \
            np.array([self.IntegratedResponseSHM(i, j, mx, fp, fn, delta)
                      for i, j in zip(self.BinEdges_left, self.BinEdges_right)])

    def RegionSHM(self, mx, fp, fn, delta, output_file):
        predicted = self._Predicted(mx, fp, fn, delta)
        sum_pred_squared = sum((predicted/self.BinError)**2)
        sum_pred_bindata = sum(predicted * self.BinData / self.BinError**2)
        sigma_fit = max(sum_pred_bindata / sum_pred_squared, 0)
        predicted *= sigma_fit
        log_likelihood_max = -0.5 * sum(((predicted - self.BinData) / self.BinError)**2)
        print("mx = ", mx)
        print("sigma_fit = ", sigma_fit)
        print("max log likelihood = ", log_likelihood_max)
        to_print = np.hstack((mx, sigma_fit, log_likelihood_max, predicted,
                              self.BinData, self.BinError))
        with open(output_file, 'ab') as f_handle:
            np.savetxt(f_handle, to_print)
        return to_print

    def UpperLimit(self, fp, fn, delta, mx_min, mx_max, num_steps, output_file,
                   processes=None):
        mx_list = np.logspace(np.log10(mx_min), np.log10(mx_max), num_steps)
        kwargs = ({'mx': mx,
                   'fp': fp,
                   'fn': fn,
                   'delta': delta,
                   'output_file': output_file}
                  for mx in mx_list)
        upper_limit = np.array(par.parmap(self.RegionSHM, kwargs, processes))
        print("mx_list = ", mx_list)
        print("upper_limit = ", upper_limit)
        return upper_limit

    def LogLMax(self, output_file, mx_fit_guess=None):
        table = np.transpose(np.loadtxt(output_file))
        mx_list = table[0]
        mx_min = mx_list[0]
        mx_max = mx_list[-1]
        if mx_fit_guess is None:
            mx_fit_guess = (mx_min + mx_max)/2
        print('mx_min, mx_max =', mx_min, mx_max)
        log_likelihood_max = table[2]
        neg_log_likelihood_max_interp = interpolate.interp1d(mx_list, -log_likelihood_max,
                                                             kind='linear')
        print('logL_guess =', neg_log_likelihood_max_interp(mx_fit_guess))
        mx_fit = minimize(neg_log_likelihood_max_interp, mx_fit_guess, tol=1e-4,
                          bounds=[(1.1 * mx_min, 0.9 * mx_max)]).x
        print('mx_fit =', mx_fit)
        logL_max = -neg_log_likelihood_max_interp(mx_fit)
        print('logL_max =', logL_max)
        return mx_fit, logL_max, mx_list, log_likelihood_max

    def _UpperLowerLists(self, mx, logL_max, sigma, pred, data, err):
        def logL(ratio, pred=pred, data=data, err=err):
            return -sum((ratio * pred - data)**2 / (2 * err**2)) - self.logL_target
        if logL_max > self.logL_target:
            ratio_low = fsolve(logL, 0.5)[0]
            ratio_high = fsolve(logL, 3)[0]
            return [[np.log10(mx), np.log10(ratio_low * sigma)],
                    [np.log10(mx), np.log10(ratio_high * sigma)]]
        return []

    def UpperLowerLists(self, CL, output_file, output_file_lower, output_file_upper,
                        num_mx=1000, processes=None):
        print('CL =', CL, 'chi2 =', chi_squared(2, CL))
        self.logL_target = self.logL_max - chi_squared(2, CL)/2

        table = np.transpose(np.loadtxt(output_file))
        num_dimensions = table.shape[0]
        num_bins = (num_dimensions - 3)//3
        print('num_dimensions =', num_dimensions)
        print('num_bins =', num_bins)
        mx_list = table[0]
        sigma_fit_interp = interpolate.interp1d(mx_list, table[1])
        log_likelihood_max_interp = interpolate.interp1d(mx_list, table[2])

        predicted_interp = [interpolate.interp1d(mx_list, table[i])
                            for i in range(3, 3 + num_bins)]
        data_interp = [interpolate.interp1d(mx_list, table[i])
                       for i in range(3 + num_bins, 3 + 2 * num_bins)]
        error_interp = [interpolate.interp1d(mx_list, table[i])
                        for i in range(3 + 2 * num_bins, 3 + 3 * num_bins)]

        mx_list = np.logspace(np.log10(mx_list[0]), np.log10(mx_list[-1]), num=num_mx)

        sigma_fit = np.array([sigma_fit_interp(mx) for mx in mx_list])
        logL_max = np.array([log_likelihood_max_interp(mx) for mx in mx_list])
        predicted = np.array([[p(mx) for p in predicted_interp] for mx in mx_list])
        data = np.array([[d(mx) for d in data_interp] for mx in mx_list])
        error = np.array([[err(mx) for err in error_interp] for mx in mx_list])

        kwargs = ({'mx': mx_list[index],
                   'logL_max': logL_max[index],
                   'sigma': sigma_fit[index],
                   'pred': predicted[index],
                   'data': data[index],
                   'err': error[index]
                   }
                  for index in range(len(mx_list)))
        limits = [l for l in par.parmap(self._UpperLowerLists, kwargs,
                                        processes=processes, verbose=False) if l]
        if limits:
            limits = np.transpose(limits, axes=(1, 0, 2))
            limit_low = limits[0]
            limit_high = limits[1]
        else:
            limit_low = limit_high = []
        return [np.array(limit_low), np.array(limit_high)]

    def Region(self, delta, CL, output_file, output_file_lower, output_file_upper,
               num_mx=1000, processes=None):
        self.mx_fit, self.logL_max, mx_list, log_likelihood_max = \
            self.LogLMax(output_file)
        [limit_low, limit_high] = \
            self.UpperLowerLists(CL, output_file, output_file_lower, output_file_upper,
                                 num_mx=num_mx, processes=processes)
        np.savetxt(output_file_lower, limit_low)
        np.savetxt(output_file_upper, limit_high)
        return


class DAMAExperimentCombined(DAMAExperiment, Experiment):
    """ Class for finding the best-fit regions for the DAMA experiment
    when considering the combined analysis of Na and I.
    Constructor:
    Input:
        exper_name: string
            Name of experiments. Must contain the list of 2 experiment names separated by
            space.
        scattering_type: string
            Type of scattering.
        mPhi: float, optional
            Mass of the mediator.
        quenching_factor: tuple, optional
            Quenching factor. If not given, the default used is specified in the data
            modules. If given, then a list or tuple of 2 quenching_factors, one for Na
            and one for I.
    """
    def __init__(self, exper_name, scattering_type, mPhi=mPhiRef, quenching_factor=None):
        exper_name = exper_name.split()
        super().__init__(exper_name[0], scattering_type, mPhi)
        self.other = self.__class__.__bases__[1](exper_name[1], scattering_type, mPhi)
        if quenching_factor is not None:
            self.QuenchingFactor = lambda e: quenching_factor[0]
            self.other.QuenchingFactor = lambda e: quenching_factor[1]

    def _Predicted(self, mx, fp, fn, delta):
        return self.Exposure * conversion_factor / mx * \
            np.array([self.IntegratedResponseSHM(i, j, mx, fp, fn, delta) +
                      self.other.IntegratedResponseSHM(i, j, mx, fp, fn, delta)
                      for i, j in zip(self.BinEdges_left, self.BinEdges_right)])

    def _exchange(self, string, old, new):
        return string.replace(old, new).replace(new, old, 1)

    def Region(self, delta, CL, output_file, output_file_lower, output_file_upper,
               num_mx=1000, processes=None, plot=False):
        if output_file.find('Na') < output_file.find('I'):
            old, new = 'I', 'Na'
        else:
            old, new = 'Na', 'I'
        flipped_file = self._exchange(output_file, old, new)
        flipped_file = self._exchange(flipped_file, str(self.other.QuenchingFactor(0)),
                                      str(self.QuenchingFactor(0)))
        self.mx_fit, self.logL_max, self.mx_list, self.log_likelihood_max = \
            self.LogLMax(output_file)
        if plot:
            plt.close()
            plt.plot(self.mx_list, self.log_likelihood_max, 'r')
            plt.plot(self.mx_list, [self.logL_max] * len(self.mx_list), 'r')

        try:
            mx_fit, logL_max, mx_list, log_likelihood_max = self.LogLMax(flipped_file)
            if plot:
                plt.plot(mx_list, log_likelihood_max, 'b')
                plt.plot(mx_list, [logL_max] * len(mx_list), 'b')
            if logL_max > self.logL_max:
                self.mx_fit = mx_fit
                self.logL_max = logL_max
                self.mx_list = mx_list
                self.log_likelihood_max = log_likelihood_max
        except:
            if plot:
                plt.plot(self.mx_list, [self.logL_max] * len(self.mx_list), 'o')
        finally:
            if plot:
                plt.show()
        [limit_low, limit_high] = \
            self.UpperLowerLists(CL, output_file, output_file_lower, output_file_upper,
                                 num_mx=num_mx, processes=processes)
#        if len(limit_low) > 0:
#            mx_separation = np.diff(limit_low[:, 0])
#            break_index = np.argmax(mx_separation)
#            if mx_separation[break_index] > 1.3 * np.average(mx_separation):
#                print('mx break =', limit_low[break_index, 0],
#                      limit_low[break_index + 1, 0])
#                if output_file.find('Na') < output_file.find('I'):
#                    limit_low = np.array(limit_low[:break_index + 1])
#                    limit_high = np.array(limit_high[:break_index + 1])
#                else:
#                    limit_low = np.array(limit_low[break_index + 1:]),
#                    limit_high = np.array(limit_high[break_index + 1:])
#        print('limit_low, limit_high =')
#        print(limit_low, limit_high)
        np.savetxt(output_file_lower, limit_low)
        np.savetxt(output_file_upper, limit_high)
        return


class DAMATotalRateExperiment(Experiment):
    """ Class for finding the upper limit due to DAMA total rate.
    Input:
        exper_name: string
            Name of experiment.
        scattering_type: string
            Type of scattering.
        mPhi: float, optional
            Mass of the mediator.
        quenching_factor: float, optional
            Quenching factor. If not given, the default used is specified in the data
            modules.
    """
    def __init__(self, exper_name, scattering_type, mPhi=mPhiRef, quenching_factor=None):
        super().__init__(exper_name, scattering_type, mPhi)
        module = import_file(INPUT_DIR + exper_name + ".py")
        self.BinEdges = module.BinEdges
        self.BinData = self.Exposure * module.BinData  # unitless
        if quenching_factor is not None:
            self.QuenchingFactor = lambda e: quenching_factor

    def RegionSHM(self, mx, fp, fn, delta, output_file):
        predicted = self.Exposure * conversion_factor / mx * \
            np.array(self.IntegratedResponseSHM(self.BinEdges[0], self.BinEdges[1],
                                                mx, fp, fn, delta))
        sigma_fit = self.BinData[0] / predicted
        print("mx = ", mx)
        print("sigma_fit = ", sigma_fit)
        to_print = np.array([[mx, sigma_fit]])
        with open(output_file, 'ab') as f_handle:
            np.savetxt(f_handle, to_print)
        return to_print.flatten()

    def UpperLimit(self, fp, fn, delta, mx_min, mx_max, num_steps, output_file,
                   processes=None):
        mx_list = np.logspace(np.log10(mx_min), np.log10(mx_max), num_steps)
        kwargs = ({'mx': mx,
                   'fp': fp,
                   'fn': fn,
                   'delta': delta,
                   'output_file': output_file}
                  for mx in mx_list)
        upper_limit = np.array(par.parmap(self.RegionSHM, kwargs, processes))
        print("mx_list = ", mx_list)
        print("upper_limit = ", upper_limit)
        return upper_limit
