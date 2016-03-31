"""
Copyright (c) 2015 Andreea Georgescu

Created on Sun Mar  1 19:51:42 2015

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
from experiment import *
from experiment_HaloIndep_er import *
from math import *
from globalfnc import *
import parallel_map as par
from scipy.linalg import det, inv
# from scipy.special import lambertw
from lambertw import *
from scipy.optimize import brentq
import os


class Experiment_HaloIndep(Experiment):
    """ Base class that implements the halo-independent analysis common to all
    experiments, using vmin as independent variable in the integration.
    Input:
        exper_name: string
            Name of experiment.
        scattering_type: string
            Type of scattering. Can be
                - 'SI' (spin-independent)
                - 'SDAV' (spin-independent, axial-vector)
                - 'SDPS' (spin-independent, pseudo-scalar)
        mPhi: float, optional
            The mass of the mediator. If not given, it corresponds to contact interaction.
    """
    def __init__(self, exper_name, scattering_type, mPhi=mPhiRef):
        Experiment.__init__(self, exper_name, scattering_type, mPhi)
        if self.energy_resolution_type == "Dirac":
            self.Response = self._Response_Dirac
        else:
            self.Response = self._Response_Finite

    def DifferentialResponse(self, Eee, qER, const_factor):
        """ Differential response function d**2 R / (d Eee d ER)
            NOT including the velocity integral eta0
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
            np.array([self.ResolutionFunction(Eee, qer, self.EnergyResolution(qer))
                      for qer in qER])
        return r_list.sum()

    def ConstFactor(self, vmin, mx, fp, fn, delta, sign):
        """ Collects the factors that don't depend on the measured energy Eee,
        so they only need to be computed once in Response function.
        Returns:
            (ER, qER, const_factor): tuple
        """

        ER = ERecoilBranch(vmin, self.mT, mx, delta, sign)
        q = self.QuenchingFactor(ER)
        qER = q * ER
        efficiencyER = self.Efficiency_ER(ER)
        const_factor = kilogram/SpeedOfLight**2 * \
            self.CrossSectionFactors(ER, mx, fp, fn, delta) * \
            np.abs(dERecoildVmin(vmin, self.mT, mx, delta, sign)) * efficiencyER
        return (ER, qER, const_factor)

    def DifferentialResponse_Full(self, vmin, Eee, mx, fp, fn, delta, sign):
        """ Differential response function d**2 R / (d Eee d ER)
            NOT including the velocity integral eta0
            Same as DifferentialResponse, but computed given full input parameters,
        instead of the pre-computed const_factor.
        """
        (ER, qER, const_factor) = self.ConstFactor(vmin, mx, fp, fn, delta, sign)
        return self.DifferentialResponse(Eee, qER, const_factor)

    def _Response_Finite(self, vmin, Eee1, Eee2, mx, fp, fn, delta):
        """ Response function integral d**2 R / (d Eee d ER) between measured energies
        Eee1 and Eee2.
            NOT including eta0.
            For any finite resolution function (i.e. other than Dirac Delta).
        """
        self.count_response_calls += 1
        if delta == 0:
            branches = [1]
        else:
            branches = [1, -1]
        result = 0
        for sign in branches:
            (ER, qER, const_factor) = self.ConstFactor(vmin, mx, fp, fn, delta, sign)
            result += integrate.quad(self.DifferentialResponse, Eee1, Eee2,
                                     args=(qER, const_factor),
                                     epsrel=PRECISSION, epsabs=0)[0]
        if result >= 0:
            return result
        return 0

    def _Response_Dirac(self, vmin, Eee1, Eee2, mx, fp, fn, delta):
        """ Response function integral d**2 R / (d Eee d ER) between measured energies
        Eee1 and Eee2.
            NOT including eta0.
            For Dirac Delta resolution function.
        """
        self.count_response_calls += 1
        if delta == 0:
            branches = [1]
        else:
            branches = [1, -1]
        r_list_sum = 0
        for sign in branches:

            ER = ERecoilBranch(vmin, self.mT, mx, delta, sign)
            q = self.QuenchingFactor(ER)
            qER = q * ER
            integrated_delta = np.array([1. if Eee1 <= i < Eee2 else 0. for i in qER])
            efficiencyEee = self.Efficiency(Eee1, qER)
#            efficiencyER = self.Efficiency_ER(qER)
            efficiencyER = np.array(list(map(self.Efficiency_ER, qER)))
            r_list = kilogram/SpeedOfLight**2 * \
                self.CrossSectionFactors(ER, mx, fp, fn, delta) * \
                np.abs(dERecoildVmin(vmin, self.mT, mx, delta, sign)) * \
                efficiencyEee * efficiencyER * integrated_delta
            r_list_sum += r_list.sum()
        return r_list_sum

    def IntegratedResponse(self, vmin1, vmin2, Eee1, Eee2, mx, fp, fn, delta):
        """ Integrated Response Function between measured energies Eee1 and Eee2,
        and all recoil energies ER.
            NOT including eta0.
            For any finite resolution function (i.e. other than Dirac Delta).
        """
        midpoints = []
        integr = integrate.quad(self.Response, vmin1, vmin2,
                                args=(Eee1, Eee2, mx, fp, fn, delta), points=midpoints,
                                epsrel=PRECISSION, epsabs=0)
        return integr[0]

    def IntegratedResponseTable(self, vmin_list, E1, E2, mx, fp, fn, delta):
        return np.array([self.IntegratedResponse(vmin_list[a], vmin_list[a+1],
                        E1, E2, mx, fp, fn, delta)
                        for a in range(0, vmin_list.size - 1)])


class MaxGapExperiment_HaloIndep(Experiment_HaloIndep):
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
        self.ElistMaxGap = np.append(np.insert(
            np.array(list(filter(lambda x: self.Ethreshold < x < self.Emaximum,
                          self.ERecoilList))),
            0, self.Ethreshold), self.Emaximum)

    def TabulateMaximumGapLimit(self, vmin1, vmin2, mx, fp, fn, delta):
        print("vmin = ", vmin2)
        return np.array(list(map(lambda i, j:
                        self.IntegratedResponse(vmin1, vmin2, i, j, mx, fp, fn, delta),
                        self.ElistMaxGap[:-1], self.ElistMaxGap[1:])))

    def MaximumGapUpperBound(self, vmin_min, vmin_max, vmin_step, mx, fp, fn, delta,
                             output_file, processes=None):
        vmin_list = np.linspace(vmin_min, vmin_max, (vmin_max - vmin_min)/vmin_step + 1)
        vmin_list0 = np.insert(vmin_list, 0, 0.)
        xtable = np.zeros(self.ElistMaxGap.size - 1)
        upperlimit_table = np.array([])
        kwargs = ({'vmin1': vmin_list0[v_index], 'vmin2': vmin_list[v_index],
                   'mx': mx, 'fp': fp, 'fn': fn, 'delta': delta}
                  for v_index in range(vmin_list.size))
        xtable_list = par.parmap(self.TabulateMaximumGapLimit, kwargs, processes)
        for v_index in range(vmin_list.size):
            xtable += xtable_list[v_index]
            mu_scaled = xtable.sum()
            x_scaled = np.max(xtable)
            if x_scaled == 0:
                mu_over_x = np.inf
                result = [np.inf]
            else:
                mu_over_x = mu_scaled / x_scaled

                y_guess = np.real(-lambertw(-0.1 / mu_over_x, -1))
                y = fsolve(lambda x: MaximumGapC0scaled(x, mu_over_x) - ConfidenceLevel,
                           y_guess)
                result = y / x_scaled / self.Exposure
                result = result[0]
                print("vmin = ", vmin_list[v_index], "   mu_over_x = ", mu_over_x)
                print("xtable = ", xtable)
                print("mu_over_x =", mu_over_x)
                print("y_guess =", y_guess)
                print("y =", y)
                print("x_scaled =", x_scaled)
                print("result = ", result)
                to_print = np.log10(np.array([[mx, result]]))
                with open(output_file, 'ab') as f_handle:
                    np.savetxt(f_handle, to_print)
            upperlimit_table = np.append(upperlimit_table, [result])
        return upperlimit_table

    def UpperLimit(self, mx, fp, fn, delta, vmin_min, vmin_max, vmin_step,
                   output_file, processes=None, **unused_kwargs):
        upper_limit = self.MaximumGapUpperBound(vmin_min, vmin_max, vmin_step, mx,
                                                fp, fn, delta, output_file,
                                                processes=processes)
        vmin_list = np.linspace(vmin_min, vmin_max, (vmin_max - vmin_min)/vmin_step + 1)
        print("vmin_list = ", vmin_list)
        print("upper_limit = ", upper_limit)
        result = np.transpose([vmin_list, np.log10(upper_limit)])
        print("res = ", result)
        return result[result[:, 1] != np.inf]


class PoissonExperiment_HaloIndep(Experiment_HaloIndep):
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

    def _PoissonUpperBound(self, vmin, mx, fp, fn, delta):
        print('vmin =', vmin)
        muT = self.mT * mx / (self.mT + mx)
        Eee_max = max(2e6 * muT**2 * (vmin/SpeedOfLight)**2 / self.mT)
        print("self.Ethreshold =", self.Ethreshold)
        print("Eee_max =", Eee_max)
        int_response = self.IntegratedResponse(0, vmin, self.Ethreshold, Eee_max,
                                               mx, fp, fn, delta)
        print("int_response =", int_response)
        if int_response > 0:
            result = np.log10(self.Expected_limit / self.Exposure / int_response)
        else:
            result = np.inf
        return [vmin, result]

    def UpperLimit(self, mx, fp, fn, delta, vmin_min, vmin_max, vmin_step,
                   output_file, processes=None, **unused_kwargs):
        vmin_list = np.linspace(vmin_min, vmin_max, (vmin_max - vmin_min)/vmin_step + 1)
        kwargs = ({'vmin': vmin, 'mx': mx, 'fp': fp, 'fn': fn, 'delta': delta}
                  for vmin in vmin_list)
        upper_limit = np.array(par.parmap(self._PoissonUpperBound, kwargs, processes))
        upper_limit = upper_limit[upper_limit[:, 1] != np.inf]
        print("upper_limit = ", upper_limit)
        with open(output_file, 'ab') as f_handle:
            np.savetxt(f_handle, upper_limit)
        return upper_limit


class GaussianExperiment_HaloIndep(Experiment_HaloIndep):
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
        self.Expected_limit = module.Expected_limit * self.BinSize
        if quenching_factor is not None:
            self.QuenchingFactor = lambda e: quenching_factor

        print('BinData', self.BinData)
        print('BinError', self.BinError)

#        If you want to calculate the limit as is done in fig 2 of arxiv 1409.5446v2
#        self.BinData = module.BinData/module.Exposure
#        self.chiSquared = chi_squared(self.BinData.size)
#        self.Expected_limit = module.Expected_limit * self.BinSize

    def _GaussianUpperBound(self, vmin, mx, fp, fn, delta):
        int_response = \
            np.array(list(map(lambda i, j:
                              self.IntegratedResponse(0, vmin, i, j, mx, fp, fn, delta),
                              self.BinEdges_left, self.BinEdges_right)))
        result = [i for i in self.Expected_limit / int_response if i > 0]
        result = np.min(result)
        if result > 0:
            result = np.log10(result)
        else:
            result = np.inf
        print("(vmin, result) =", (vmin, result))
        return [vmin, result]

    def UpperLimit(self, mx, fp, fn, delta, vmin_min, vmin_max, vmin_step,
                   output_file, processes=None, **unused_kwargs):
        vmin_list = np.linspace(vmin_min, vmin_max, (vmin_max - vmin_min)/vmin_step + 1)
        kwargs = ({'vmin': vmin, 'mx': mx, 'fp': fp, 'fn': fn, 'delta': delta}
                  for vmin in vmin_list)
        upper_limit = np.array(par.parmap(self._GaussianUpperBound, kwargs, processes))
        upper_limit = upper_limit[upper_limit[:, 1] != np.inf]
        print("upper_limit = ", upper_limit)
        with open(output_file, 'ab') as f_handle:
            np.savetxt(f_handle, upper_limit)
        return upper_limit

    def _MinusLogLikelihood(self, vars_list, mx, fp, fn, delta,
                            vminStar=None, logetaStar=None, vminStar_index=None):
        """ Compute -log(L)
        Input:
            vars_list: ndarray
                List of variables [vmin_1, ..., vmin_No, log(eta_1), ..., log(eta_No)]
            vminStar, logetaStar: float, optional
                Values of fixed vmin^* and log(eta)^*.
        Returns:
            -2log(L): float

        """

        if vminStar is None:
            vmin_list_w0 = vars_list[: vars_list.size/2]
            logeta_list = vars_list[vars_list.size/2:]
        else:
            vmin_list_w0 = np.insert(vars_list[: vars_list.size/2],
                                     vminStar_index, vminStar)
            logeta_list = np.insert(vars_list[vars_list.size/2:],
                                    vminStar_index, logetaStar)
        vmin_list_w0 = np.insert(vmin_list_w0, 0, 0)

        rate_partials = [None] * (self.BinEdges_left.size)

        for x in range(0, self.BinEdges_left.size):
            resp_integr = self.IntegratedResponseTable(vmin_list_w0,
                                                       self.BinEdges_left[x],
                                                       self.BinEdges_right[x],
                                                       mx, fp, fn, delta)

            rate_partials[x] = np.dot(10**logeta_list, resp_integr)

        result = 0

        if self.energy_resolution_type == "Dirac":
            self.Response = self._Response_Dirac
        else:
            self.Response = self._Response_Finite

        for x in range(0, self.BinData.size):
                if self.BinData[x] != 0:
                    result += ((rate_partials[x] - self.BinData[x] / self.Exposure) ** 2.0 /
                               (self.BinError[x] ** 2.0))

        return result

    def KKT_Condition_Q(self, vars_list, mx, fp, fn, delta,
                        vminStar=None, logetaStar=None, vminStar_index=None):
        """This is intended to calculate the contribution to q(vmin) from a particular experiment.
        It may currently have issues, has not been tested.
        """

        if vminStar is None:
            vmin_list_w0 = vars_list[: vars_list.size/2]
            logeta_list = vars_list[vars_list.size/2:]

        vmin_list_w0 = np.insert(vmin_list_w0, 0, 0)

        self.curly_H_tab = np.zeros((self.BinData.size, 1001))
        self.Q_contrib = np.zeros((self.BinData.size, 1001))
        rate_partials = [None] * (self.BinEdges_left.size)

        for x in range(0, self.BinEdges_left.size):
            resp_integr = self.IntegratedResponseTable(vmin_list_w0,
                                                       self.BinEdges_left[x],
                                                       self.BinEdges_right[x],
                                                       mx, fp, fn, delta)

            rate_partials[x] = np.dot(10**logeta_list, resp_integr)

        if self.energy_resolution_type == "Dirac":
            self.Response = self._Response_Dirac
        else:
            self.Response = self._Response_Finite

        result = np.zeros((self.BinData.size, 1001))
        # TODO parallelize this section of the code
        calculate_Q = False
        if calculate_Q:
            for x in range(0, self.BinData.size):
                for v_dummy in range(1, 1001):
                    self.curly_H_tab[x, v_dummy] = integrate.quad(self.Response, min(VminDelta(self.mT, mx, delta)), v_dummy,
                                                                  args=(self.BinEdges_left[x], self.BinEdges_right[x], mx,
                                                                        fp, fn, delta), epsrel=PRECISSION, epsabs=0)[0]
                    if self.BinData[x] != 0:
                            result[x, v_dummy] = (2.0 * ((rate_partials[x] - self.BinData[x] / self.Exposure) /
                                                         self.BinError[x] ** 2.0) * self.curly_H_tab[x, v_dummy])

                    self.Q_contrib[x, v_dummy] = result[x, v_dummy]
            file = Output_file_name(self.name, self.scattering_type, self.mPhi, mx, fp, fn, delta,
                                    F, "_KKT_Cond_1", "../Output_Band/") + ".dat"
            f_handle = open(file, 'wb')   # clear the file first
            np.savetxt(f_handle, self.Q_contrib)
            f_handle.close()
        else:
            file = Output_file_name(self.name, self.scattering_type, self.mPhi, mx, fp, fn, delta,
                                    F, "_KKT_Cond_1", "../Output_Band/") + ".dat"
            f_handle = open(file, 'rb')
            self.Q_contrib = np.loadtxt(f_handle)
            f_handle.close()

        print('Obtained Variational of Likelihood')

        return self.Q_contrib


class Poisson_Likelihood(Experiment_HaloIndep):
    """ Class for multi EHI method experiments with binned possoin analysis.
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
        self.BinExposure = module.BinExposure

        self.Expected_limit = module.Expected_limit / self.BinExposure

        self.mT_avg = np.sum(module.target_nuclide_AZC_list[:, 2] * module.target_nuclide_mass_list)

        print('BinData', self.BinData)

    def UpperLimit(self, mx, fp, fn, delta, vmin_min, vmin_max, vmin_step,
                   output_file, processes=None, **unused_kwargs):
        vmin_list = np.linspace(vmin_min, vmin_max, (vmin_max - vmin_min)/vmin_step + 1)
        kwargs = ({'vmin': vmin, 'mx': mx, 'fp': fp, 'fn': fn, 'delta': delta}
                  for vmin in vmin_list)
        upper_limit = np.array(par.parmap(self._PoissonUpperBound, kwargs, processes))
        upper_limit = upper_limit[upper_limit[:, 1] != np.inf]
        print("upper_limit = ", upper_limit)
        with open(output_file, 'ab') as f_handle:
            np.savetxt(f_handle, upper_limit)
        return upper_limit

    def _PoissonUpperBound(self, vmin, mx, fp, fn, delta):
        int_response = \
            np.array(list(map(lambda i, j:
                              self.IntegratedResponse(0, vmin, i, j, mx, fp, fn, delta),
                              self.BinEdges_left, self.BinEdges_right)))
        result = [i for i in self.Expected_limit / int_response if i > 0]
        result = np.min(result)
        if result > 0:
            result = np.log10(result)
        else:
            result = np.inf
        print("(vmin, result) =", (vmin, result))
        return [vmin, result]

    def _MinusLogLikelihood(self, vars_list, mx, fp, fn, delta,
                            vminStar=None, logetaStar=None, vminStar_index=None):
        """ Compute -log(L)
        Input:
            vars_list: ndarray
                List of variables [vmin_1, ..., vmin_No, log(eta_1), ..., log(eta_No)]
            vminStar, logetaStar: float, optional
                Values of fixed vmin^* and log(eta)^*.
        Returns:
            -2log(L): float

        """

        if vminStar is None:
            vmin_list_w0 = vars_list[: vars_list.size/2]
            logeta_list = vars_list[vars_list.size/2:]
        else:
            vmin_list_w0 = np.insert(vars_list[: vars_list.size/2],
                                     vminStar_index, vminStar)
            logeta_list = np.insert(vars_list[vars_list.size/2:],
                                    vminStar_index, logetaStar)

        vmin_list_w0 = np.insert(vmin_list_w0, 0, 0)

        rate_partials = [None] * (self.BinEdges_left.size)

        for x in range(0, self.BinEdges_left.size):
            resp_integr = self.IntegratedResponseTable(vmin_list_w0,
                                                       self.BinEdges_left[x],
                                                       self.BinEdges_right[x],
                                                       mx, fp, fn, delta)

            rate_partials[x] = np.dot(10**logeta_list, resp_integr)

        result = 0

        if self.energy_resolution_type == "Dirac":
            self.Response = self._Response_Dirac
        else:
            self.Response = self._Response_Finite

        for x in range(0, self.BinData.size):
            result += 2.0 * (self.BinExposure[x] * rate_partials[x] + self.BinBkgr[x] + log(factorial(self.BinData[x])) -
                             self.BinData[x] * log(self.BinExposure[x] * rate_partials[x] + self.BinBkgr[x]))

        return result

    def KKT_Condition_Q(self, vars_list, mx, fp, fn, delta,
                        vminStar=None, logetaStar=None, vminStar_index=None):
        """This is intended to calculate the contribution to q(vmin) from a particular experiment.
        It may currently have issues, has not been tested.
        """

        if vminStar is None:
            vmin_list_w0 = vars_list[: vars_list.size/2]
            logeta_list = vars_list[vars_list.size/2:]
        else:
            vmin_list_w0 = np.insert(vars_list[: vars_list.size/2],
                                     vminStar_index, vminStar)
            logeta_list = np.insert(vars_list[vars_list.size/2:],
                                    vminStar_index, logetaStar)

        vmin_list_w0 = np.insert(vmin_list_w0, 0, 0)

        self.curly_H_tab = np.zeros((self.BinData.size, 1001))
        self.Q_contrib = np.zeros((self.BinData.size, 1001))
        rate_partials = [None] * (self.BinEdges_left.size)

        for x in range(0, self.BinEdges_left.size):
            resp_integr = self.IntegratedResponseTable(vmin_list_w0,
                                                       self.BinEdges_left[x],
                                                       self.BinEdges_right[x],
                                                       mx, fp, fn, delta)

            rate_partials[x] = np.dot(10**logeta_list, resp_integr)

        if self.energy_resolution_type == "Dirac":
            self.Response = self._Response_Dirac
        else:
            self.Response = self._Response_Finite

        result = np.zeros((self.BinData.size, 1001))
        # TODO parallelize this section of the code
        calculate_Q = False
        if calculate_Q:
            for x in range(0, self.BinData.size):
                for v_dummy in range(1, 1001):
                        self.curly_H_tab[x, v_dummy] = integrate.quad(self.Response, min(VminDelta(self.mT, mx, delta)), v_dummy,
                                                                      args=(self.BinEdges_left[x], self.BinEdges_right[x], mx,
                                                                            fp, fn, delta), epsrel=PRECISSION, epsabs=0)[0]

                        result[x, v_dummy] = (2.0 * ((self.BinExposure[x] * rate_partials[x] + self.BinBkgr[x] - self.BinData[x]) /
                                                     (rate_partials[x] + self.BinBkgr[x] / self.BinExposure[x])) *
                                              self.curly_H_tab[x, v_dummy])
                        self.Q_contrib[x, v_dummy] = result[x, v_dummy]

            file = Output_file_name(self.name, self.scattering_type, self.mPhi, mx, fp, fn, delta,
                                    F, "_KKT_Cond_1", "../Output_Band/") + ".dat"
            f_handle = open(file, 'wb')   # clear the file first
            np.savetxt(f_handle, self.Q_contrib)
            f_handle.close()
        else:
            file = Output_file_name(self.name, self.scattering_type, self.mPhi, mx, fp, fn, delta,
                                    F, "_KKT_Cond_1", "../Output_Band/") + ".dat"
            f_handle = open(file, 'rb')
            self.Q_contrib = np.loadtxt(f_handle)
            f_handle.close()

        print('Obtained Variational of Likelihood')

        return self.Q_contrib


class Crosses_HaloIndep(Experiment_HaloIndep):
    """ Class for finding the crosses for experients with potential signal and
    binned data.
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
        self.BinEdges_left = self.BinEdges[:-1]
        self.BinEdges_right = self.BinEdges[1:]
        self.BinData = module.BinData
        self.BinError = module.BinError
        self.QuenchingFactorOfEee = module.QuenchingFactorOfEee
        if quenching_factor is not None:
            self.QuenchingFactor = lambda e: quenching_factor
        self._int_resp = self.IntegratedResponse

    def _VminRange(self, E1, E2, mT, mx, delta):
        E_delta = - delta * mx / (mT + mx)
        vmin_of_E1 = VMin(E1, mT, mx, delta)
        vmin_of_E2 = VMin(E2, mT, mx, delta)
        print(vmin_of_E1, vmin_of_E2)
        if E1 <= E_delta and E2 >= E_delta:
            vmin_min = 0
        else:
            vmin_min = min(vmin_of_E1, vmin_of_E2)
        vmin_max = max(vmin_of_E1, vmin_of_E2)
        return (vmin_min, vmin_max)

    def _AverageOverNuclides(self, quantity):
        return np.sum(quantity * self.mass_fraction) / np.sum(self.mass_fraction)

    def _Box(self, Eee1, Eee2, mT_avg, mx, fp, fn, delta, vmax, output_file=None):
        print(self.name)
        print('Eee1 =', Eee1, ' Eee2 =', Eee2)
        dvmin = 1
        if delta <= 0:
            vmin_list = np.linspace(0, vmax, (vmax + dvmin)/dvmin)
            resp_list = [self.Response(vmin1, Eee1, Eee2, mx, fp, fn, delta)
                         for vmin1 in vmin_list[:-1]] + [0.001]
        else:
            vdelta = min(VminDelta(self.mT, mx, delta))
            print('vdelta =', vdelta)
            vdelta = max(0, vdelta // dvmin * dvmin - dvmin)
            vmin_list = np.linspace(vdelta, vmax, (vmax - vdelta + dvmin)/dvmin)
            resp_list = [self.IntegratedResponse(vmin1, vmin2, Eee1, Eee2,
                                                 mx, fp, fn, delta)/dvmin
                         for vmin1, vmin2 in zip(vmin_list[:-1], vmin_list[1:])] + [0.001]

        plt.close()
        plt.plot(vmin_list, resp_list, '-')
        int_resp = sum(resp_list) * dvmin
        index_center = np.argmax(resp_list)
        vmin_center = vmin_list[index_center]
        resp_max = resp_list[index_center]
        resp_min = max(resp_list[0], resp_list[-1])

        if output_file is not None:
            output_file = output_file.replace('temp.dat', self.name + '_' + str(Eee1) +
                                              '_' + str(Eee2) + '.dat')
            print(output_file)
            np.savetxt(output_file,
                       np.transpose([vmin_list, np.array(resp_list)/int_resp]))
            output_file = output_file.replace('.dat', '_notnorm.dat')
            print(output_file)
            np.savetxt(output_file, np.transpose([vmin_list, resp_list]))

        if index_center > 0:
            int_resp_left = \
                interpolate.interp1d(resp_list[index_center::-1],
                                     dvmin * np.cumsum(resp_list[index_center::-1]))
        else:
            def int_resp_left(r): return 0
        if index_center < len(resp_list) - 1:
            int_resp_right = \
                interpolate.interp1d(resp_list[index_center:],
                                     dvmin * np.cumsum(resp_list[index_center:]))
        else:
            def int_resp_right(r): return 0

        print('resp_max =', resp_max)
        print('resp_min =', resp_min)
        print('int_resp =', int_resp)

        def integrated_response(r):
            return int_resp_left(r) + int_resp_right(r) - resp_max -\
                        ConfidenceLevel * int_resp

        print(integrated_response(resp_min * 1.1), integrated_response(resp_max * 0.9))

        response_CL = brentq(integrated_response, resp_min * 1.1, resp_max * 0.9)
        print('response_CL =', response_CL)
        plt.plot(vmin_list, response_CL * np.ones_like(vmin_list), '-')

        vmin_interp_left = interpolate.interp1d(resp_list[:index_center + 1],
                                                vmin_list[:index_center + 1])
        vmin_interp_right = interpolate.interp1d(resp_list[index_center:],
                                                 vmin_list[index_center:])
        vmin_error_left = - vmin_interp_left(response_CL) + vmin_center
        vmin_error_right = vmin_interp_right(response_CL) - vmin_center
        print('vmin_edges =',
              VMin(Eee1/self.QuenchingFactor(Eee1), self.mT, mx, delta)[0],
              VMin(Eee2/self.QuenchingFactor(Eee2), self.mT, mx, delta)[0])
        print('vmin_interp =', vmin_interp_left(response_CL),
              vmin_interp_right(response_CL))
        print('vmin_center =', vmin_center)
        print('vmin_error =', vmin_error_left, vmin_error_right)

#        os.system("say 'Plot'")
#        plt.show()

        return (int_resp, vmin_center, vmin_error_left, vmin_error_right)

    def _Boxes(self, mx, fp, fn, delta, vmax=2000, processes=None, output_file=None):
        mT_avg = np.sum(self.mT * self.mass_fraction) / np.sum(self.mass_fraction)
        print("mT_avg =", mT_avg)
        print('vmax =', vmax)
        arraybox = np.array([])
        for Eee1, Eee2 in zip(self.BinEdges_left, self.BinEdges_right):
            kwargs = {'Eee1': Eee1, 'Eee2': Eee2, 'mT_avg': mT_avg,
                      'mx': mx, 'fp': fp, 'fn': fn, 'delta': delta, 'vmax': vmax,
                      'output_file': output_file}
            if Eee1 == self.BinEdges_left[0]:
                arraybox = np.append(arraybox, self._Box(**kwargs))
            else:
                arraybox = np.column_stack((arraybox, np.array(self._Box(**kwargs))))

        return arraybox

    def _Rebin(self, index=9):
        self.BinEdges = np.append(self.BinEdges[:index + 1],  self.BinEdges[-1])
        data, error = Rebin_data(self.BinData[index:], self.BinError[index:])
        self.BinData = np.append(self.BinData[:index], data)
        self.BinError = np.append(self.BinError[:index], error)
        print('BinEdges =', self.BinEdges)
        print('BinData =', self.BinData)
        print('BinError =', self.BinError)
        self.BinEdges_left = self.BinEdges[:-1]
        self.BinEdges_right = self.BinEdges[1:]

    def UpperLimit(self, mx, fp, fn, delta, vmin_min, vmin_max, vmin_step,
                   output_file, rebin=False, processes=None, **unused_kwargs):
        if rebin:
            self._Rebin()

        box_table = self._Boxes(mx, fp, fn, delta, vmax=vmin_max, processes=processes)

        int_resp_list = box_table[0, :]
        vmin_center_list = box_table[1, :]
        vmin_error_left_list = box_table[2, :]
        vmin_error_right_list = box_table[3, :]
        eta_list = self.BinData / int_resp_list
        eta_error_list = self.BinError / int_resp_list
        print('Bin Data', self.BinData)
        print('Bin Error', self.BinError)
        print('eta error', eta_list)
        print('eta error list', eta_error_list)
        result = np.array([int_resp_list, vmin_center_list, vmin_error_left_list,
                           vmin_error_right_list, eta_list, eta_error_list])
        print(result)
        with open(output_file, 'ab') as f_handle:
            np.savetxt(f_handle, result)
        return result

    def IntResponseMatrix(self, mx, fp, fn, delta, vmin_min, vmin_max, vmin_step,
                          output_file, processes=None):
        np.set_printoptions(threshold=np.nan)
        vmin_list = np.linspace(vmin_min, vmin_max, (vmin_max - vmin_min)/vmin_step + 1)
        kwargs = ({'vmin1': vmin1, 'vmin2': vmin2,
                   'Eee1': Eee1, 'Eee2': Eee2,
                   'mx': mx, 'fp': fp, 'fn': fn, 'delta': delta}
                  for vmin1, vmin2 in zip(vmin_list[:-1], vmin_list[1:])
                  for Eee1, Eee2 in zip(self.BinEdges_left, self.BinEdges_right))
        matr = [self._int_resp(**k) for k in kwargs]
        matr = np.reshape(matr, (len(vmin_list)-1, len(self.BinEdges_left)))
        print('matrix =')
        print(matr)
        print(np.shape(matr))
        print('determinant =', det(matr))
        print('inverse =')
        print(inv(matr))

        with open(output_file, 'ab') as f_handle:
            np.savetxt(f_handle, matr)
        return matr


class Crosses_HaloIndep_Combined(Crosses_HaloIndep, Experiment_HaloIndep):
    """ Class for finding the best-fit regions for the DAMA experiment
    when considering the combined analysis of Na and I.
    Constructor:
        A list or tuple of 2 experiment names must be given, and, if not None, then
    a list or tuple of 2 quenching_factors, one for Na and one for I.
    """
    def __init__(self, exper_name, scattering_type, mPhi=mPhiRef, quenching_factor=None):
        exper_name = exper_name.split()
        super().__init__(exper_name[0], scattering_type, mPhi)
        self.other = self.__class__.__bases__[0](exper_name[1], scattering_type, mPhi)
        if quenching_factor is not None:
            self.QuenchingFactor = lambda e: quenching_factor[0]
            self.other.QuenchingFactor = lambda e: quenching_factor[1]

    def _int_resp(self, vmin1, vmin2, Eee1, Eee2, mx, fp, fn, delta):
        return self.IntegratedResponse(vmin1, vmin2, Eee1, Eee2, mx, fp, fn, delta) \
            + self.other.IntegratedResponse(vmin1, vmin2, Eee1, Eee2, mx, fp, fn, delta)

    def _Rebin(self, initial_energy_bin, vmax, mx, num_rebinned_bins=19):
        # build the new self.BinEdges_left and self.BinEdges_right
        self.BinEdges_left = [initial_energy_bin[0]]
        self.BinEdges_right = [initial_energy_bin[1]]
        ratio = ERecoil_ratio(self.mT, self.other.mT, mx,
                              self.QuenchingFactor(0), self.other.QuenchingFactor(0))
        ratio = round(ratio[0], 1)
        print('ratio =', ratio)
        vmin_left_edge = VMin(self.BinEdges_left[-1]/self.QuenchingFactor(0),
                              self.mT[0], mx, 0)
        print('vmax =', vmax)
        print('vmin_left_edge =', vmin_left_edge)
        while vmin_left_edge < vmax:
            self.BinEdges_left.append(self.BinEdges_left[-1] * ratio)
            self.BinEdges_right.append(self.BinEdges_right[-1] * ratio)
            vmin_left_edge = VMin(self.BinEdges_left[-1]/self.QuenchingFactor(0),
                                  self.mT[0], mx, 0)
            print('vmin_left_edge =', vmin_left_edge)
        self.other.BinEdges_left = self.BinEdges_left
        self.other.BinEdges_right = self.BinEdges_right
        print('BinEdges_left =', self.BinEdges_left)
        print('BinEdges_right =', self.BinEdges_right)

        if self.BinEdges_right[-1] > self.BinEdges[-1]:
            # add fake bins at higher energies
            index = len(self.BinData) - num_rebinned_bins
            data, error = Rebin_data(self.BinData[index:], self.BinError[index:])
            num_added_bins = round((self.BinEdges_right[-1] - self.BinEdges[-1]) /
                                   (self.BinEdges[-1] - self.BinEdges[-2]))
            added_edges = np.linspace(self.BinEdges[-1], self.BinEdges_right[-1],
                                      num_added_bins + 1)
            self.BinEdges = np.append(self.BinEdges, added_edges)
            self.BinData = np.append(self.BinData,
                                     [data/num_rebinned_bins] * num_added_bins)
            self.BinError = np.append(self.BinError,
                                      [error/np.sqrt(num_rebinned_bins)] * num_added_bins)
            print('BinEdges =', self.BinEdges)
            print('BinData =', self.BinData)
            print('BinError =', self.BinError)

        # combine multiple bins to fit the edges from self.BinEdges_left and _right
        self.BinData_rebinned = []
        self.BinError_rebinned = []
        for index in range(len(self.BinEdges_left)):
            data = np.array([d for i, d in enumerate(self.BinData)
                             if self.BinEdges[i] >= self.BinEdges_left[index] and
                             self.BinEdges[i + 1] <= self.BinEdges_right[index]])
            error = np.array([d for i, d in enumerate(self.BinError)
                              if self.BinEdges[i] >= self.BinEdges_left[index] and
                              self.BinEdges[i + 1] <= self.BinEdges_right[index]])
            print('data =', data)
            print('error =', error)
            data_rebinned, error_rebinned = Rebin_data(data, error)
            self.BinData_rebinned.append(data_rebinned)
            self.BinError_rebinned.append(error_rebinned)
        print('BinData_rebinned =', self.BinData_rebinned)
        print('BinError_rebinned =', self.BinError_rebinned)

    def UpperLimit(self, mx, fp, fn, delta, vmin_min, vmin_max, vmin_step,
                   output_file, initial_energy_bin=[2, 4], vmax=None, processes=None,
                   **unused_kwargs):
        if delta != 0:
            raise ValueError('delta has to be zero for DAMA halo-independent ' +
                             'combined analysis!')
        if vmax is None:
            vmax = vmin_step
        self._Rebin(initial_energy_bin, vmax, mx)
        box_table = self._Boxes(mx, fp, fn, delta, vmax=vmin_max, processes=processes,
                                output_file=output_file)
        box_table_other = self.other._Boxes(mx, fp, fn, delta, vmax=vmin_max,
                                            processes=processes, output_file=output_file)
        print('box_table =')
        print(repr(box_table))
        print('box_table_other =')
        print(repr(box_table_other))
        int_resp_list = box_table[:, 0]
        int_resp_list_other = box_table_other[:, 0]
        vmin_center_list = box_table_other[:, 1]
        vmin_error_left_list = box_table_other[:, 2]
        vmin_error_right_list = box_table_other[:, 3]
        size = len(int_resp_list)
        int_resp_matrix = np.vstack((np.hstack((np.zeros((size - 1, 1)),
                                               np.diag(int_resp_list[:-1]))),
                                     np.zeros(size)))
        int_resp_matrix += np.diag(int_resp_list_other)
        print('int_resp_matrix =', int_resp_matrix)
        int_resp_inverse = np.linalg.inv(int_resp_matrix)

        eta_list = np.dot(int_resp_inverse, self.BinData_rebinned)
        eta_error_list = np.sqrt(np.dot(int_resp_inverse ** 2,
                                        np.array(self.BinError_rebinned) ** 2))
        result = np.array([int_resp_list + int_resp_list_other,
                           vmin_center_list, vmin_error_left_list, vmin_error_right_list,
                           eta_list, eta_error_list])
        print(result)
        with open(output_file, 'ab') as f_handle:
            np.savetxt(f_handle, result)
        return result


class Standard_Halo_Model:
    def __init__(self, exper_name, log_sigma_p):
        self.name = exper_name
        self.log_sigma_p = log_sigma_p

    def UpperLimit(self, mx, fp, fn, delta, vmin_min, vmin_max, vmin_step,
                   output_file, **unused_kwargs):
        if "eta0" in self.name:
            eta = eta0Maxwellian
        else:
            eta = eta1Maxwellian
        vmin_list = np.linspace(vmin_min, vmin_max, (vmin_max - vmin_min)/vmin_step + 1)
        # print('vmin_list =', vmin_list)
        eta_list = eta(vmin_list, vobs, v0bar, vesc)
        eta_list = np.array([i if i > 0 else np.inf for i in eta_list])
        # print('eta_list =', eta_list)
        log_eta_list = self.log_sigma_p + np.log10(conversion_factor / mx * eta_list)
        # print('log_eta_list =', log_eta_list)
        result = np.transpose([vmin_list, log_eta_list])
        result = result[result[:, 1] != np.inf]
        print(result)
        with open(output_file, 'ab') as f_handle:
            np.savetxt(f_handle, result)
        return result
