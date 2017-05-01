"""
Copyright (c) 2015 Andreea Georgescu

Created on Wed Mar  4 00:47:37 2015

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

# TODO! This only works for CDMSSi!

from __future__ import division
from __future__ import absolute_import
from __future__ import print_function
from experiment_HaloIndep import *
import interp_uniform as unif
# from interp import interp1d
from scipy import interpolate
from scipy.optimize import brentq, minimize, brute
from basinhopping import *
from globalfnc import *
from custom_minimizer import *
import matplotlib.pyplot as plt
import os   # for speaking
import parallel_map as par
from scipy.stats import poisson
import numpy.random as random
import numpy as np
from scipy.interpolate import interp1d
from scipy.integrate import quad
import copy

DEBUG = F
DEBUG_FULL = F
USE_BASINHOPPING = F
ADAPT_KWARGS = F
ALLOW_MOVE = T


class ConstraintsFunction(object):
    """ Class to implement the constraints function that will be passed as an argunent
    to the minimization routines.
    Input:
        args: Arguments needed for calculating the constraints:
            vminStar, logetaStar, vminStar_index
    """
    def __init__(self, *args):
        self.vminStar = args[0]
        self.logetaStar = args[1]
        self.vminStar_index = args[2]
        self.vmin_max = 2000

    def __call__(self, x, close=True):
        """
        Input:
            x: ndarray
        Returns:
            constraints: ndarray
                Constraints vector, where each value must be >= 0 for the
                constraint to be specified. Contains:
             0 -  8: bounds: 3 * (x.size/2) constraints = 9 for x.size/2 = 3
             9 - 12: sorted array: 2 * (x.size/2 - 1) constraints = 4 for x.size/2 = 3
            13 - 15: vminStar_index: x.size/2 constraints = 3 for x.size/2 = 3
            16 - 18: vminStar and logetaStar: x.size/2 constraints = 3 for x.size/2 = 3
        """
        hxsz = int(x.size/2)
        constraints = np.concatenate([x[:hxsz], self.vmin_max - x[:hxsz], -x[hxsz:],
                                      np.diff(x[:hxsz]), np.diff(-x[hxsz:]),
                                      (x[:hxsz] - self.vminStar) * (-x[hxsz:] + self.logetaStar),
                                      self.vminStar - x[:self.vminStar_index],
                                      x[self.vminStar_index: hxsz] - self.vminStar,
                                      x[hxsz: hxsz + self.vminStar_index] - self.logetaStar,
                                      self.logetaStar - x[hxsz + self.vminStar_index:]])

        if close:
            is_not_close = np.logical_not(np.isclose(constraints, np.zeros_like(constraints), atol=1e-5))
            is_not_close[:3 * hxsz] = True
            constraints = np.where(is_not_close, constraints, np.abs(constraints))
        if np.any(np.isnan(constraints)):
            raise ValueError

        return constraints


class Experiment_EHI(Experiment_HaloIndep):
    """ Class implementing the extended maximum likelihood halo-independent (EHI)
    method to obtain the confidence band for experiments with potential signals and
    unbinned data (arXiv:1507.03902).
    Input:
        expername: string
            The name of the experiment.
        scattering_type: string
            The type of scattering. Can be
                - 'SI' (spin-independent)
                - 'SDAV' (spin-dependent, axial-vector)
                - 'SDPS' (spin-dependent, pseudo-scalar)
        mPhi: float, optional
            The mass of the mediator.
        method: str, optional
            Type of minimization solver to be passed as a parameter to the minimization
                routine. Can be 'SLSQP' or 'COBYLA'.
    """
    def __init__(self, expername, scattering_type, mPhi=mPhiRef, method='SLSQP', pois=False, gaus=False):
        super().__init__(expername, scattering_type, mPhi)
        module = import_file(INPUT_DIR + expername + ".py")
        if not pois and not gaus:
            self.ERecoilList = module.ERecoilList
            self.mu_BKG_i = module.mu_BKG_i
            self.NBKG = module.NBKG
            self.method = method
            self.mu_BKG_interp = interp1d(self.ERecoilList, self.mu_BKG_i, bounds_error=False)
            self.Poisson = False
            self.Gaussian = False
        elif pois:
            self.BinData = module.BinData
            self.BinEdges_l = module.BinEdges_left
            self.BinEdges_r = module.BinEdges_right
            self.Binbkg = module.BinBkgr
            self.BinExp = module.BinExposure
            self.Poisson = True
            self.Nbins = module.Nbins


    def _VMinSortedList(self, mx, fp, fn, delta):
        """ Computes the list of vmin corresponsing to measured recoil energies,
        sorted in increasing order. Will be useful as starting guesses.
        """

        if not self.Poisson and not self.Gaussian:
            self.vmin_sorted_list = np.sort(VMin(self.ERecoilList, self.mT[0], mx, delta))
        else:
            self.vmin_sorted_list = np.sort(VMin(self.BinEdges_r, self.mT[-1], mx, delta))
        return

    def ResponseTables(self, vmin_min, vmin_max, vmin_step, mx, fp, fn, delta,
                       output_file_tail):
        """ Computes response tables
            - self.diff_response_tab is a table of [vmin, DifferentialResponse(Eee_i)]
        pairs for each vmin in the range [vminmin, vminmax], corresponding to measured
        recoil energies Eee_i. It is a 3D matrix where
                axis = 0 has dimension self.ERecoilList.size()
                axis = 1 has dimension vmin_list.size() + 1 (where + 1 is because we
                    prepend zeros for vmin = 0)
                axis = 2 has dimension 2 for the pairs of [vmin, diff_response].
            - self.response_tab is a table of [vmin, Response] pairs for each vmin
        in the range [vminmin, vminmax], corresponding to DifferentialResponse
        integrated over the full energy range. It is a 2D matrix where
                axis = 1 has dimension vmin_list.size() + 1 (where +1 is because we
                    prepend zeros for vmin = 0)
                axis = 2 has dimension 2 for the pairs of [vmin, diff_response].
        Input:
            vmin_min, vmin_max, vmin_step: float
                Vmin range and vmin step size.
            mx, fp, fn, delta: float
            output_file_tail: string
                Tag to be added to the file name since the results for
                self.vmin_sorted_list, self.diff_response_tab and self.response_tab
                are each written to files.
        """

        self._VMinSortedList(mx, fp, fn, delta)
        file = output_file_tail + "_VminSortedList.dat"
        print(file)
        np.savetxt(file, self.vmin_sorted_list)

        if delta == 0:
            branches = [1]
        else:
            branches = [1, -1]
        self.vmin_linspace = np.linspace(vmin_min, vmin_max, (vmin_max - vmin_min)/vmin_step + 1)

        if not self.Poisson and not self.Gaussian:
            self.diff_response_tab = np.zeros((self.ERecoilList.size, 1))
            self.curly_H_tab = np.zeros((self.ERecoilList.size, 1))
            self.response_tab = np.zeros(1)
            self.xi_tab = np.zeros(1)

            xi = 0
            vmin_prev = 0
            for vmin in self.vmin_linspace:
                print("vmin =", vmin)
                resp = 0
                diff_resp_list = np.zeros((1, len(self.ERecoilList)))
                curly_H = np.zeros((1, len(self.ERecoilList)))
                for sign in branches:
                    (ER, qER, const_factor) = self.ConstFactor(vmin, mx, fp, fn, delta, sign)
                    v_delta = min(VminDelta(self.mT, mx, delta))
                    diff_resp_list += np.array([self.DifferentialResponse(Eee, qER, const_factor)
                                                for Eee in self.ERecoilList])
                    curly_H += np.array([[integrate.quad(self.DifferentialResponse_Full, v_delta, vmin,
                                                         args=(Eee, mx, fp, fn, delta, sign),
                                                         epsrel=PRECISSION, epsabs=0)[0]
                                          for Eee in self.ERecoilList]])

                    resp += integrate.quad(self.DifferentialResponse, self.Ethreshold, self.Emaximum,
                                           args=(qER, const_factor), epsrel=PRECISSION, epsabs=0)[0]

                xi += self.Exposure * \
                    self.IntegratedResponse(vmin_prev, vmin,
                                            self.Ethreshold, self.Emaximum,
                                            mx, fp, fn, delta)
                vmin_prev = vmin
                self.diff_response_tab = \
                    np.append(self.diff_response_tab, diff_resp_list.transpose(), axis=1)
                self.response_tab = np.append(self.response_tab, [resp], axis=0)
                self.curly_H_tab = np.append(self.curly_H_tab, curly_H.transpose(), axis=1)
                # counts/kg/keVee
                self.xi_tab = np.append(self.xi_tab, [xi], axis=0)
                # counts * day
            file = output_file_tail + "_DiffRespTable.dat"
            print(file)
            np.savetxt(file, self.diff_response_tab)
            file = output_file_tail + "_RespTable.dat"
            print(file)
            np.savetxt(file, self.response_tab)
            file = output_file_tail + "_CurlyHTable.dat"
            print(file)
            np.savetxt(file, self.curly_H_tab)
            file = output_file_tail + "_XiTable.dat"
            print(file)
            np.savetxt(file, self.xi_tab)
        elif self.Poisson:
            self.response_tab = np.zeros((len(self.vmin_linspace), len(self.BinEdges_l)))
            self.xi_tab = np.zeros((len(self.vmin_linspace), len(self.BinEdges_l)))
            v_delta = min(VminDelta(self.mT, mx, delta))
            vmin_prev = 0
            xi = np.zeros(len(self.BinData))
            for i, vmin in enumerate(self.vmin_linspace):
                print("vmin =", vmin)
                for j in range(len(self.BinEdges_r)):
                    for sign in branches:
                        (ER, qER, const_factor) = self.ConstFactor(vmin, mx, fp, fn, delta, sign)

                        # self.response_tab[i, j] += self._Response_Dirac(vmin, self.BinEdges_l[j],
                        #                                                 self.BinEdges_r[j], mx, fp, fn, delta)
                        self.response_tab[i, j] += self._Response_Finite(vmin, self.BinEdges_l[j],
                                                                          self.BinEdges_r[j], mx, fp, fn, delta)
                        #print(vmin, ERecoilBranch(vmin, 120., mx, delta, 1.), self.BinEdges_l[j], self.response_tab[i, j])

                    xi[j] += self.Exposure * np.trapz(self.response_tab[:i,j], self.vmin_linspace[:i])

                    vmin_prev = vmin

                    self.xi_tab[i, j] = xi[j]
            self.minVmin = self.vmin_linspace[np.argmin(self.response_tab[:, 0])]
            self.response_tab = np.insert(self.response_tab, 0, 0, axis=0)
            self.xi_tab = np.insert(self.xi_tab, 0, 0, axis=0)

            file = output_file_tail + "_RespTable.dat"
            print(file)
            np.savetxt(file, self.response_tab)
            file = output_file_tail + "_XiTable.dat"
            print(file)
            np.savetxt(file, self.xi_tab)


        self.vmin_linspace = np.insert(self.vmin_linspace, 0, 0)
        file = output_file_tail + "_VminLinspace.dat"
        print(file)
        np.savetxt(file, self.vmin_linspace)

#        os.system("say Finished response tables.")
        return

    def PlotTable(self, func, dimension=0, xlim=None, ylim=None,
                  title=None, plot_close=True, plot_show=True, show_zero_axis=False):
        """ Plots response tables.
        Input:
            func: callable
                Function or list of functions of v that should be plotted.
            dimension: int
                0 (if there's only one function) or
                1 (if there are a list of functions).
            xlim, ylim: float
                Axis limits for the plots.
            title: string
                Plot title.
            plot_close, plot_show: bool
                Whether to call plt.close() before and plt.show() after.
            show_zero_axis: bool
                Whether to show a horizontal line at zero.
        """
        if plot_close:
            plt.close()
        if dimension == 0:
            # only one function
            plt.plot(self.vmin_linspace, np.array([func(v)
                     for v in self.vmin_linspace]))
        elif dimension == 1:
            # list of interpolated functions for each energy in self.ERecoilList
            for i in range(self.ERecoilList.size):
                plt.plot(self.vmin_linspace, np.array([func[i](v)
                         for v in self.vmin_linspace]))
        else:
            print("Wrong dimension")
            raise TypeError
        if show_zero_axis:
            plt.plot(self.vmin_linspace, np.zeros(self.vmin_linspace.size))
        if xlim is not None:
            plt.xlim(xlim)
        if ylim is not None:
            plt.ylim(ylim)
        if title is not None:
            plt.title(title)
        if plot_show:
            plt.show()

    def ImportResponseTables(self, output_file_tail, plot=False, pois=False):
        """ Imports the data for the response tables from files.
        """
        file = output_file_tail + "_VminSortedList.dat"
        with open(file, 'r') as f_handle:
            self.vmin_sorted_list = np.loadtxt(f_handle)
        file = output_file_tail + "_VminLinspace.dat"
        with open(file, 'r') as f_handle:
            self.vmin_linspace = np.loadtxt(f_handle)
        file = output_file_tail + "_RespTable.dat"
        with open(file, 'r') as f_handle:
            self.response_tab = np.loadtxt(f_handle)
        file = output_file_tail + "_XiTable.dat"
        with open(file, 'r') as f_handle:
            self.xi_tab = np.loadtxt(f_handle)

        if not self.Poisson and not self.Gaussian:
            file = output_file_tail + "_DiffRespTable.dat"
            with open(file, 'r') as f_handle:
                self.diff_response_tab = np.loadtxt(f_handle)

            file = output_file_tail + "_CurlyHTable.dat"
            with open(file, 'r') as f_handle:
                self.curly_H_tab = np.loadtxt(f_handle)

            self.diff_response_interp = np.array([unif.interp1d(self.vmin_linspace, dr)
                                                  for dr in self.diff_response_tab])
            self.curly_H_interp = np.array([unif.interp1d(self.vmin_linspace, h)
                                            for h in self.curly_H_tab])

            self.response_interp = unif.interp1d(self.vmin_linspace, self.response_tab)
        else:
            self.response_interp = [None] * len(self.response_tab[1,:])
            self.xi_interp = [None] * len(self.xi_tab[1, :])
            for i in range(len(self.response_tab[1,:])):
                self.response_interp[i] = unif.interp1d(self.vmin_linspace, self.response_tab[:, i])
                self.xi_interp[i] = interp1d(self.vmin_linspace, self.xi_tab[:, i])
            tab = next(i for i,x in enumerate(self.response_tab[:, 0]) if x > 0.)
            self.minVmin = self.vmin_linspace[tab]
        if plot:
            self.PlotTable(self.diff_response_interp, dimension=1)
            self.PlotTable(self.response_interp, dimension=0)
            self.PlotTable(self.curly_H_interp, dimension=1, title='Curly H')
        return

    def VminIntegratedResponseTable(self, vmin_list):
        vmin_max = 1200.
        vmin_list[vmin_list > vmin_max] = vmin_max
        tab = np.zeros((vmin_list.size-1) * self.ERecoilList.size)
        tab = tab.reshape((self.ERecoilList.size, vmin_list.size-1))

        for a in range(vmin_list.size - 1):
            for i in range(self.ERecoilList.size):
                if (vmin_list[a+1] - vmin_list[a]) > 1.0 and (vmin_list[a+1] < 1000.):
                    tab[i,a] = integrate.quad(self.diff_response_interp[i],
                                      vmin_list[a], vmin_list[a + 1],
                                      epsrel=PRECISSION, epsabs=0)[0]

        return tab

    def IntegratedResponseTable(self, vmin_list, i=-1):
        if self.Poisson:
            response_interp = self.response_interp[i]
        else:
            response_interp = self.response_interp
        tab = np.zeros(vmin_list.size - 1)
        for a in range(vmin_list.size - 1):
            if (vmin_list[a+1] - vmin_list[a]) > 0.1 and (vmin_list[a+1] < 1000.):
                tab[a] = integrate.quad(response_interp,
                                        vmin_list[a], vmin_list[a + 1],
                                        epsrel=PRECISSION, epsabs=0)[0]

        return tab


    def diffRespPois(self, vmin, i=-1):
        return self.response_interp[i](vmin)


    def ExpectedNumEvents(self, minfunc, mx, fp, fn, delta):

        vmin_list_w0 = minfunc[:(minfunc.size / 2)]
        logeta_list = minfunc[(minfunc.size / 2):]
        vmin_list_w0 = np.insert(vmin_list_w0, 0, 0)

        resp_integr = self.IntegratedResponseTable(vmin_list_w0)
        Nsignal = self.Exposure * np.dot(10**logeta_list, resp_integr)

        return Nsignal

    def Simulate_Events(self, Nexpected, minfunc, class_name, mx, fp, fn, delta):

        Nevents = poisson.rvs(Nexpected + 0.41)

        vdelta=min(VminDelta(self.mT, mx, delta))
        logeta_list = minfunc[(minfunc.size / 2):]
        eta_list = np.insert(logeta_list,0,-1)
        vmin_list_w0 = minfunc[:(minfunc.size / 2)]
        vmin_list_w0 = np.insert(vmin_list_w0, vdelta, 0)
        vmin_grid = np.linspace(vdelta, vmin_list_w0[-1], 1000)

        x_run = 0
        resp_integr = np.zeros(len(vmin_grid))
        for vmin_ind in range(len(vmin_grid)):
            if vmin_grid[vmin_ind] < (vmin_list_w0[x_run+1]):
                resp_integr[vmin_ind] = 10**eta_list[x_run] * self.response_interp(vmin_grid[vmin_ind])
            else:
                x_run+=1
                resp_integr[vmin_ind] = 10**eta_list[x_run] * self.response_interp(vmin_grid[vmin_ind])

        #TODO This needs to be generalized for the future. MC must be done in ER
        #for inelastic scattering and translated to vmin

        if Nevents > 0:

            pdf = resp_integr / np.sum(resp_integr)

            cdf = pdf.cumsum()
            u = random.rand(Nevents)
            Q = np.zeros(Nevents)
            for i in np.arange(Nevents):
                Q[i] = vmin_grid[np.absolute(cdf - u[i]).argmin()]
            Q = np.sort(Q)
        else:
            Q = np.array([])
            Nevents = 0


        print('Events expected: ', (Nexpected + 0.41), 'Events Simulated: ', Nevents)
        print('Events: ', Q)
        for x in Q:
             recoil = ERecoilBranch(x, self.mT[0], mx, delta, 1)
             if x == Q[0]:
                 self.ERecoilList = np.array([recoil])
             else:
                 self.ERecoilList = np.append(self.ERecoilList,recoil)

        self.ERecoilList = np.sort(self.ERecoilList)
        for x in range(len(self.ERecoilList)):
            if self.ERecoilList[x] < self.Ethreshold:
                self.ERecoilList[x] = self.Ethreshold + .01
            elif self.ERecoilList[x] > self.Emaximum:
                self.ERecoilList[x] = self.Emaximum - .01

        print('Recoil List', self.ERecoilList)

        self.vmin_linspace = np.linspace(vdelta, 1000, 800)

        self.diff_response_tab = np.zeros((self.ERecoilList.size, 1))

        self.mu_BKG_i = np.zeros(len(self.ERecoilList))

        for x in range(0, len(self.ERecoilList)):
            self.mu_BKG_i[x] = 0.7

        for vmin in self.vmin_linspace:
            diff_resp_list = np.zeros((1, len(self.ERecoilList)))

            if delta==0:
                branches = [1]
            else:
                branches = [1,-1]
            for sign in branches:
                (ER, qER, const_factor) = self.ConstFactor(vmin, mx, fp, fn, delta, sign)
                diff_resp_list += np.array([self.DifferentialResponse(Eee, qER, const_factor)
                                            for Eee in self.ERecoilList])


            self.diff_response_tab = np.append(self.diff_response_tab, diff_resp_list.transpose(), axis=1)
            self.diff_response_interp = np.array([unif.interp1d(self.vmin_linspace, dr)
                                                  for dr in self.diff_response_tab])

        return Q

    def _MinusLogLikelihood(self, vars_list, vminStar=None, logetaStar=None,
                            vminStar_index=None):
        """ Compute -log(L)
        Input:
            vars_list: ndarray
                List of variables [vmin_1, ..., vmin_No, log(eta_1), ..., log(eta_No)]
            vminStar, logetaStar: float, optional
                Values of fixed vmin^* and log(eta)^*.
        Returns:
            -log(L): float
        """

        if vminStar is None:
            vmin_list_w0 = vars_list[: int(vars_list.size/2)]
            logeta_list = vars_list[int(vars_list.size/2):]
        else:
            vmin_list_w0 = np.insert(vars_list[: int(vars_list.size/2)],
                                     vminStar_index, vminStar)
            logeta_list = np.insert(vars_list[int(vars_list.size/2):],
                                    vminStar_index, logetaStar)

        vmin_list_w0 = np.insert(vmin_list_w0, 0, 0)

        if not self.Poisson:
            resp_integr = self.IntegratedResponseTable(vmin_list_w0)
            vmin_resp_integr = self.VminIntegratedResponseTable(vmin_list_w0)
            mu_i = self.Exposure * np.dot(vmin_resp_integr, 10**logeta_list)
            Nsignal = self.Exposure * np.dot(10**logeta_list, resp_integr)
            if Nsignal < 0.:
                Nsignal = 0.

            if vminStar is None:
                self.gamma_i = (self.mu_BKG_i + mu_i) / self.Exposure
                # counts/kg/keVee/days
            for x in range(0, len(mu_i)):
                if mu_i[x] <= 0.:
                    mu_i[x] = 0.
            result = 2.0 * (self.NBKG + Nsignal - np.log(self.mu_BKG_i + mu_i).sum())
        else:
            result = 0.
            for i in range(len(self.BinData)):
                resp_integr = self.IntegratedResponseTable(vmin_list_w0, i=i)
                bin_ev = self.BinExp[i] * np.dot(10 ** logeta_list, resp_integr)
                tot_ev = (bin_ev + self.Binbkg[i])

                result += 2.0 * (tot_ev - self.BinData[i] * np.log(tot_ev))

        return result


    def MinusLogLikelihood(self, vars_list, constr_func=None, vminStar=None,
                           logetaStar=None, vminStar_index=None):
        """ Computes -log(L) and tests whether constraints are satisfied.
        Input:
            vars_list: ndarray
                List of variables [vmin_1, ..., vmin_No, log(eta_1), ..., log(eta_No)].
            constr_func: callable, optional
                Ffunction of vars_list giving an array of values each corresponding to
                a constraint. If the values are > 0 the constraints are satisfied.
            vminStar, logetaStar: float, optional
                Values of fixed vmin^* and log(eta)^*.
            vminStar_index: int, optional
                Index corresponding to the position of vminStar in the array of vmin
                steps.
        Returns:
            -log(L) if all constraints are valid, and the result of an artificial
                function that grows with the invalid constraints if not all constraints
                are valid.
        """
        constraints = constr_func(vars_list)
        constr_not_valid = constraints < 0

        if DEBUG_FULL:
            print("*** vars_list =", repr(vars_list))
        if DEBUG_FULL:
            print("vminStar =", vminStar)
            print("logetaStar =", logetaStar)
            print("constraints =", repr(constraints))
            print("constr_not_valid =", repr(constr_not_valid))
        try:
            return self._MinusLogLikelihood(vars_list, vminStar=vminStar,
                                            logetaStar=logetaStar,
                                            vminStar_index=vminStar_index)
        except:
            if np.any(constr_not_valid):
                constr_list = constraints[constr_not_valid]
                if DEBUG_FULL:
                    print("Constraints not valid!!")
                    print("constr sum =", -constr_list.sum())
                return min(max(-constr_list.sum(), 0.001) * 1e6, 1e6)
            else:
                print("Error!!")
                raise

    def OptimalLikelihood(self, output_file_tail, logeta_guess, pois=False, gaus=False):
        """ Finds the best-fit piecewise constant eta function corresponding to the
        minimum MinusLogLikelihood, and prints the results to file (value of the minimum
        MinusLogLikelihood and the corresponding values of vmin, logeta steps.
        Input:
            output_file_tail: string
                Tag to be added to the file name.
            logeta_guess: float
                Guess for the value of log(eta) in the minimization procedure.
        """
        self.ImportResponseTables(output_file_tail, plot=False)
        vars_guess = np.append(self.vmin_sorted_list,
                               logeta_guess * np.ones(self.vmin_sorted_list.size))
        print("vars_guess =", vars_guess)
        vmin_max = self.vmin_linspace[-1]

        def constr_func(x, vmin_max=vmin_max):
            """ 0 -  8: bounds: 3 * (x.size/2) constraints = 9 for x.size/2 = 3
                9 - 12: sorted array: 2 * (x.size/2 - 1) constraints = 4 for x.size/2 = 3
            """
            constraints = np.concatenate([x[:x.size/2], vmin_max - x[:x.size/2],
                                          -x[x.size/2:],
                                          np.diff(x[:x.size/2]), np.diff(-x[x.size/2:])])
            is_not_close = np.logical_not(
                np.isclose(constraints, np.zeros_like(constraints), atol=1e-5))
            is_not_close[:3 * (x.size/2)] = T
            constr = np.where(is_not_close, constraints, np.abs(constraints))
            if DEBUG:
                print("***constr =", repr(constr))
                print("tf =", repr(constr < 0))
            return constr
        constr = ({'type': 'ineq', 'fun': constr_func})

        np.random.seed(0)
        if USE_BASINHOPPING:
            minimizer_kwargs = {"constraints": constr, "args": (constr_func,)}
            optimum_log_likelihood = basinhopping(self.MinusLogLikelihood, vars_guess,
                                                  minimizer_kwargs=minimizer_kwargs,
                                                  niter=30, stepsize=0.1)
        else:
            optimum_log_likelihood = minimize(self.MinusLogLikelihood, vars_guess,
                                              args=(constr_func,), constraints=constr)

        #print(optimum_log_likelihood)
        print("MinusLogLikelihood =", self._MinusLogLikelihood(optimum_log_likelihood.x))
        print("vars_guess =", repr(vars_guess))
        file = output_file_tail + "_GloballyOptimalLikelihood.dat"
        print(file)
        np.savetxt(file, np.append([optimum_log_likelihood.fun],
                                   optimum_log_likelihood.x))
#        os.system("say 'Finished finding optimum'")
        return

    def ImportOptimalLikelihood(self, output_file_tail, plot=False):
        """ Import the minumum -log(L) and the locations of the steps in the best-fit
        logeta function.
        Input:
            output_file_tail: string
                Tag to be added to the file name.
            plot: bool, optional
                Whether to plot response tables.
        """
        self.ImportResponseTables(output_file_tail, plot=False)
        file = output_file_tail + "_GloballyOptimalLikelihood.dat"
        with open(file, 'r') as f_handle:
            optimal_result = np.loadtxt(f_handle)
        self.optimal_logL = optimal_result[0]
        self.optimal_vmin = optimal_result[1: optimal_result.size/2 + 1]
        self.optimal_logeta = optimal_result[optimal_result.size/2 + 1:]
        print("optimal result =", optimal_result)

        if plot:
            self._MinusLogLikelihood(optimal_result[1:])  # to get self.gamma_i
            self.xi_interp = unif.interp1d(self.vmin_linspace, self.xi_tab)
            self.h_sum_tab = np.sum([self.curly_H_tab[i] / self.gamma_i[i]
                                     for i in range(self.optimal_vmin.size)], axis=0)
            self.q_tab = 2 * (self.xi_tab - self.h_sum_tab)
            self.h_sum_interp = unif.interp1d(self.vmin_linspace, self.h_sum_tab)
            self.q_interp = unif.interp1d(self.vmin_linspace, self.q_tab)
            print('gamma', self.gamma_i[0], self.gamma_i[1], self.gamma_i[2])
            file = output_file_tail + "_HSumTable.dat"
            print(file)
            np.savetxt(file, self.h_sum_tab)
            file = output_file_tail + "_QTable.dat"
            print(file)
            np.savetxt(file, self.q_tab)

            self.PlotTable(self.xi_interp, dimension=0, plot_show=False)
            self.PlotTable(self.h_sum_interp, dimension=0,
                           xlim=[0, 2000], ylim=[-2e24, 2e24],
                           title='Xi, H_sum', plot_close=False)
            self.PlotTable(self.q_interp, dimension=0,
                           xlim=[0, 2000], ylim=[-2e24, 2e24],
                           title='q', show_zero_axis=True)
        return

    def _PlotStepFunction(self, vmin_list, logeta_list,
                          xlim_percentage=(0., 1.1), ylim_percentage=(1.01, 0.99),
                          mark=None, color=None, linewidth=1,
                          plot_close=True, plot_show=True):
        """ Plots a step-like function, given the location of the steps.
        """
        if plot_close:
            plt.close()
        print(vmin_list)
        print(logeta_list)
        x = np.append(np.insert(vmin_list, 0, 0), vmin_list[-1] + 0.1)
        y = np.append(np.insert(logeta_list, 0, logeta_list[0]), -80)
        if color is not None:
            plt.step(x, y, color=color, linewidth=linewidth)
            if mark is not None:
                plt.plot(x, y, mark, color=color)
        else:
            plt.step(x, y, linewidth=linewidth)
            if mark is not None:
                plt.plot(x, y, mark)
#        plt.xlim([vmin_list[0] * xlim_percentage[0], vmin_list[-1] * xlim_percentage[1]])
        plt.xlim([0, 1000])
        plt.ylim([max(logeta_list[-1] * ylim_percentage[0], -60),
                  max(logeta_list[0] * ylim_percentage[1], -35)])
        if plot_show:
            plt.show()
        return

    def MultiExperimentMinusLogLikelihood(self, vars_list, multiexper_input, class_name,
                                          mx, fp, fn, delta, constr_func=None, vminStar=None,
                                          logetaStar=None, vminStar_index=None):
        """ Computes -log(L) and tests whether constraints are satisfied.
        Input:
            vars_list: ndarray
                List of variables [vmin_1, ..., vmin_No, log(eta_1), ..., log(eta_No)].
            constr_func: callable, optional
                Ffunction of vars_list giving an array of values each corresponding to
                a constraint. If the values are > 0 the constraints are satisfied.
            vminStar, logetaStar: float, optional
                Values of fixed vmin^* and log(eta)^*.
            vminStar_index: int, optional
                Index corresponding to the position of vminStar in the array of vmin
                steps.
        Returns:
            -log(L) if all constraints are valid, and the result of an artificial
                function that grows with the invalid constraints if not all constraints
                are valid.
        """

        constraints = constr_func(vars_list)
        constr_not_valid = constraints < 0
        expernum = multiexper_input.size
        if DEBUG_FULL:
            print("*** vars_list =", repr(vars_list))
        if DEBUG_FULL:
            print("vminStar =", vminStar)
            print("logetaStar =", logetaStar)
            print("constraints =", repr(constraints))
            print("constr_not_valid =", repr(constr_not_valid))
        try:
            if not self.Poisson and not self.Gaussian:
                return (self._MinusLogLikelihood(vars_list, vminStar=vminStar,
                                                logetaStar=logetaStar,
                                                 vminStar_index=vminStar_index) +
                       sum([class_name[x]._MinusLogLikelihood(vars_list,
                                                             mx, fp, fn, delta)
                         for x in range(1, expernum)]))
            elif self.Poisson:
                (self._MinusLogLikelihood(vars_list, vminStar=vminStar,
                                          logetaStar=logetaStar,
                                          vminStar_index=vminStar_index) +
                 sum([class_name[x]._MinusLogLikelihood(vars_list,
                                                        mx, fp, fn, delta)
                      for x in range(1, expernum)]))
        except:
            if np.any(constr_not_valid):
                constr_list = constraints[constr_not_valid]
                if DEBUG_FULL:
                    print("Constraints not valid!!")
                    print("constr sum =", -constr_list.sum())
                return min(max(-constr_list.sum(), 0.001) * 1e6, 1e6)
            else:
                print("Error!!")
                raise

    def MultiExperimentOptimalLikelihood(self, multiexper_input, class_name, mx, fp, fn,
                                         delta, output_file_tail, output_file_CDMS, logeta_guess, nsteps_bin):
        """ Finds the best-fit piecewise constant eta function corresponding to the
        minimum MinusLogLikelihood, and prints the results to file (value of the minimum
        MinusLogLikelihood and the corresponding values of vmin, logeta steps.
        Input:
            output_file_tail: string
                Tag to be added to the file name.
            logeta_guess: float
                Guess for the value of log(eta) in the minimization procedure.
        """

        self.ImportResponseTables(output_file_CDMS, plot=False)

        if not self.Poisson:
            addsteps = self.vmin_sorted_list[-1]* np.ones(nsteps_bin)
            vminhold = np.append(self.vmin_sorted_list,addsteps)
            vmin_list = np.sort(vminhold)
        else:
            vminhold = np.array([self.vmin_sorted_list])
            for i in range(0, len(class_name)):
                if i == 0:
                    continue
                vminhold = np.append(vminhold, class_name[i]._VMinSortedList(mx, fp, fn, delta))
            print(vminhold)
            vmin_list = np.sort(vminhold)

        vars_guess = np.append(vmin_list, logeta_guess * np.ones(vmin_list.size))
        print("vars_guess = ", vars_guess)

        vmin_max = self.vmin_linspace[-1]



        if not self.Poisson:
            constr = ({'type': 'ineq', 'fun': self.constr_func})
            optimum_log_likelihood, fun_val = \
                    Custom_SelfConsistent_Minimization(class_name, vars_guess, mx, fp, fn, delta)

        else:

            logeta_bnd = (-40.0, -12.0)
            bnd_eta = [logeta_bnd] * int(vars_guess.size / 2)
            vmin_bnd = (0, vmin_max)
            bnd_vmin = [vmin_bnd] * int(vars_guess.size / 2)
            bnd = bnd_vmin + bnd_eta

            constr = ({'type': 'ineq', 'fun': self.constr_func})

            opt = minimize(self.poisson_wrapper, vars_guess,
                           args=(class_name, mx, fp, fn, delta),
                           jac=self.pois_jac,
                           method='SLSQP', bounds=bnd, constraints=constr, tol=1e-4,
                           options={'maxiter':200, 'disp':False})

            optimum_log_likelihood = opt.x
            fun_val = opt.fun

        print(optimum_log_likelihood)
        print("MinusLogLikelihood =", fun_val)
        print("vars_guess =", repr(vars_guess))
        file = output_file_tail + "_GloballyOptimalLikelihood.dat"
        print(file)
        np.savetxt(file, np.append([fun_val], optimum_log_likelihood))
        return

    def constr_func(self, x, vmin_max=1000.):
        """ 0 -  8: bounds: 3 * (x.size/2) constraints = 9 for x.size/2 = 3
            9 - 12: sorted array: 2 * (x.size/2 - 1) constraints = 4 for x.size/2 = 3
        """
        constraints = np.concatenate([x[:int(x.size/2)], vmin_max - x[:int(x.size/2)],
                                      -x[int(x.size/2):],
                                      np.diff(x[:int(x.size/2)]), np.diff(-x[int(x.size/2):])])

        is_not_close = np.logical_not(
            np.isclose(constraints, np.zeros_like(constraints), atol=1e-3))
        is_not_close[:3 * int(x.size/2)] = T
        constr = np.where(is_not_close, constraints, np.abs(constraints))
        if DEBUG:
            print("***constr =", repr(constr))
            print("tf =", repr(constr < 0))
        return constr


    def pois_jac(self, x0, class_name, mx, fp, fn, delta,
                 vminStar=None, logetaStar=None, vminStar_index=None):


        vmin_l = x0[:int(x0.size/2)]
        eta_l = x0[int(x0.size/2):]

        if vminStar is not None:
            vmin_l = np.insert(vmin_l, vminStar_index, vminStar)
            eta_l = np.insert(eta_l, vminStar_index, logetaStar)

        vmin_list_w0 = np.insert(vmin_l, 0, 0)
        eta_l_w0 = np.append(eta_l, np.array([-100.]))

        dm_deta = np.zeros(len(eta_l))
        dm_dv = np.zeros(len(vmin_l))


        for cname in class_name:
            resp_integr = np.zeros(len(cname.BinData) * len(eta_l)).reshape((len(cname.BinData),
                                                                                    len(eta_l)))
            npre = np.zeros(len(cname.BinData))
            for i in range(len(cname.BinData)):
                resp_integr[i] = cname.IntegratedResponseTable(vmin_list_w0, i=i)
                npre[i] = cname.BinExp[i] * np.dot(10 ** eta_l, resp_integr[i])
            coef = 2. * (1. - cname.BinData / (cname.Binbkg + npre))

            for i in range(len(vmin_l)):
                for j in range(int(cname.Nbins)):
                    dm_deta[i] += cname.Exposure * np.log(10.) * 10 ** eta_l[i] * coef[j] * resp_integr[j, i]
                    dm_dv[i] += coef[j] * cname.Exposure * \
                                self.diffRespPois(vmin_l[i], i=j)*(10**eta_l_w0[i] - 10**eta_l_w0[i+1])
        if vminStar is not None:
            dm_deta = np.delete(dm_deta, vminStar_index)
            dm_dv = np.delete(dm_dv, vminStar_index)

        ret = np.concatenate((dm_dv, dm_deta))
        return ret


    def poisson_wrapper(self, x0, class_name, mx, fp, fn, delta,
                        vminStar=None, logetaStar=None, index=None):
        vmin_max = 1000.


        if vminStar is not None:
            vmin_list_reduced = x0[: int(len(x0) / 2)]
            vmin_list = np.sort(np.append(vmin_list_reduced, vminStar))

            index_hold = np.argwhere(vmin_list == vminStar)[0, 0]
            logeta_list_reduced = x0[int(len(x0) / 2):]
            logeta_list = np.insert(logeta_list_reduced, index_hold, logetaStar)
        else:
            vmin_list = x0[: int(len(x0) / 2)]
            logeta_list = x0[int(len(x0) / 2):]
            logeta_list_reduced = logeta_list
            vmin_list_reduced = vmin_list
            index_hold = None
        #print(vmin_list, logeta_list)
        optimize_func = self._MinusLogLikelihood(np.append(vmin_list, logeta_list), vminStar, logetaStar, index_hold)

        for i in range(1, len(class_name)):
            try:
                optimize_func += class_name[i]._MinusLogLikelihood(np.append(vmin_list, logeta_list),
                                                                   mx, fp, fn, delta, vminStar,
                                                                   logetaStar, index_hold)
            except:
                optimize_func += class_name[i]._MinusLogLikelihood(np.append(vmin_list, logeta_list),
                                                                   vminStar, logetaStar, index_hold)
        return optimize_func

    def PlotQ_KKT_Multi(self, class_name, mx, fp, fn, delta, output_file, plot=False):

        vminlist = class_name[0].optimal_vmin
        logetalist = class_name[0].optimal_logeta
        print(vminlist, logetalist)
        optimum_steps = np.concatenate((vminlist, logetalist))
        print('optimum_steps', optimum_steps)
        explen = len(class_name)

        if plot:
            Q_contrib = [None] * (explen)
            for x in range(1, explen):
                Q_contrib[x] = class_name[x].KKT_Condition_Q(optimum_steps, mx, fp, fn, delta)

            q_sum = np.zeros(1001)

            for x in range(0, 1000):
                for y in range(1, explen):
                    q_sum[x + 1] = Q_contrib[y][:, x].sum()
            q_tab = q_sum
            if not self.Poisson:
                class_name[0]._MinusLogLikelihood(optimum_steps)  # to get self.gamma_i
                xi_interp = unif.interp1d(class_name[0].vmin_linspace, class_name[0].xi_tab)
                h_sum_tab = np.sum([class_name[0].curly_H_tab[i] / class_name[0].gamma_i[i]
                                    for i in range(self.ERecoilList.size)], axis=0)
                print('gamma', class_name[0].gamma_i[0], class_name[0].gamma_i[1], class_name[0].gamma_i[2])
                q_tab += 2.0 * (class_name[0].xi_tab - h_sum_tab)
                self.h_sum_interp = unif.interp1d(self.vmin_linspace, h_sum_tab)
                self.PlotTable(self.h_sum_interp, dimension=0,
                               xlim=[0, 2000], ylim=[-2e24, 5e25],
                               title='Xi, H_{sum}', plot_close=False)


            else:
                vmin_list = optimum_steps[:int(len(optimum_steps)/2)]
                logeta_list = optimum_steps[int(len(optimum_steps)/2):]
                vmin_list_w0 = np.insert(vmin_list, 0, 0)
                print(vmin_list_w0, logeta_list)
                xi_interp = self.xi_interp[0]
                prefac = np.zeros(len(self.BinData))
                for i in range(len(self.BinData)):
                    resp_integr = self.IntegratedResponseTable(vmin_list_w0, i=i)
                    bin_ev = self.BinExp[i] * np.dot(10 ** logeta_list, resp_integr)
                    #print(bin_ev)
                    tot_ev = (bin_ev + self.Binbkg[i])
                    #print('Pre-factor in Bin ', i+1,' of ', len(self.BinData), ' is: ', 1. - self.BinData[i] / tot_ev)
                    prefac[i] = 1. - self.BinData[i] / tot_ev
                    q_tab += (1. - self.BinData[i] / tot_ev) * self.xi_tab[:, i]
            print('Prefactors: ', prefac)
            if np.any(np.abs(prefac) < 1e-3):
                print('BF Not Unique!')
                self.uniqueBF = False
            file = output_file + "KKT_Q.dat"
            f_handle = open(file, 'wb')   # clear the file first
            np.savetxt(f_handle, q_tab)
            f_handle.close()

            self.q_interp = unif.interp1d(self.vmin_linspace, q_tab)
            # self.PlotTable(xi_interp, dimension=0, plot_show=False)
            # self.PlotTable(self.q_interp, dimension=0,
            #                xlim=[0, 1000], ylim=[-2e24, 5e26],
            #                title='q', show_zero_axis=True)

        return

    def ImportMultiOptimalLikelihood(self, output_file_tail, output_file_CDMS, plot=False):
        """ Import the minumum -log(L) and the locations of the steps in the best-fit
        logeta function.
        Input:
            output_file_tail: string
                Tag to be added to the file name.
            plot: bool, optional
                Whether to plot response tables.
        """
        self.ImportResponseTables(output_file_CDMS, plot=False)
        file = output_file_tail + "_GloballyOptimalLikelihood.dat"
        with open(file, 'r') as f_handle:
            optimal_result = np.loadtxt(f_handle)
        self.optimal_logL = optimal_result[0]
        self.optimal_vmin = optimal_result[1: int(optimal_result.size/2) + 1]
        self.optimal_logeta = optimal_result[int(optimal_result.size/2) + 1:]
        print("optimal result =", optimal_result)

        if plot:
            self._MinusLogLikelihood(optimal_result[1:])  # to get self.gamma_i
            self.xi_interp = unif.interp1d(self.vmin_linspace, self.xi_tab)
            self.h_sum_tab = np.sum([self.curly_H_tab[i] / self.gamma_i[i]
                                     for i in range(self.optimal_vmin.size)], axis=0)
            self.q_tab = 2 * (self.xi_tab - self.h_sum_tab)
            self.h_sum_interp = unif.interp1d(self.vmin_linspace, self.h_sum_tab)
            self.q_interp = unif.interp1d(self.vmin_linspace, self.q_tab)

            file = output_file_tail + "_HSumTable.dat"
            print(file)
            np.savetxt(file, self.h_sum_tab)
            file = output_file_tail + "_QTable.dat"
            print(file)
            np.savetxt(file, self.q_tab)

            self.PlotTable(self.xi_interp, dimension=0, plot_show=False)
            self.PlotTable(self.h_sum_interp, dimension=0,
                           xlim=[0, 2000], ylim=[-2e24, 2e24],
                           title='Xi, H_sum', plot_close=False)
            self.PlotTable(self.q_interp, dimension=0,
                           xlim=[0, 2000], ylim=[-2e24, 2e24],
                           title='q', show_zero_axis=True)
        return

    def PlotOptimum(self, xlim_percentage=(0., 1.1), ylim_percentage=(1.01, 0.99),
                    color='red', linewidth=1,
                    plot_close=True, plot_show=True):
        """ Plots the best-fit eta(vmin) step function.
        """
        self._PlotStepFunction(self.optimal_vmin, self.optimal_logeta,
                               xlim_percentage=xlim_percentage,
                               ylim_percentage=ylim_percentage,
                               color=color, linewidth=linewidth,
                               plot_close=plot_close, plot_show=plot_show)
        return

    def PlotConstrainedOptimum(self, vminStar, logetaStar, vminStar_index,
                               xlim_percentage=(0., 1.1), ylim_percentage=(1.01, 0.99),
                               plot_close=True, plot_show=True):
        """ Plots the eta(vmin) function given the location of vminStar and logetaStar.
        """
        self._PlotStepFunction(self.optimal_vmin, self.optimal_logeta,
                               plot_close=plot_close, plot_show=False)
        x = np.insert(self.constr_optimal_vmin, vminStar_index, vminStar)
        y = np.insert(self.constr_optimal_logeta, vminStar_index, logetaStar)
        self._PlotStepFunction(x, y,
                               xlim_percentage=xlim_percentage,
                               ylim_percentage=ylim_percentage,
                               plot_close=False, plot_show=False, mark='x', color='k')
        plt.plot(vminStar, logetaStar, '*')
        if plot_show:
            plt.show()
        return

    def _ConstrainedOptimalLikelihood(self, vminStar, logetaStar, vminStar_index):
        """ Finds the constrained minimum MinusLogLikelihood for given vminStar,
        logetaStar and vminStar_index.
        Input:
            vminStar, logetaStar: float
                Location of the constrained step.
            vminStar_index: int
                Index of vminStar in the list of vmin steps of the constrained optimum
                logeta function.
        Returns:
            constr_optimal_logl: float
                The constrained minimum MinusLogLikelihood
        """
        if DEBUG:
            print("~~~~~ vminStar_index =", vminStar_index)
        vmin_guess_left = np.array([self.optimal_vmin[ind]
                                    if self.optimal_vmin[ind] < vminStar
                                    else vminStar * (1 - 0.001*(vminStar_index - ind))
                                    for ind in range(vminStar_index)])
        vmin_guess_right = np.array([self.optimal_vmin[ind]
                                    if self.optimal_vmin[ind] > vminStar
                                    else vminStar * (1 + 0.001*(ind - vminStar_index - 1))
                                    for ind in range(vminStar_index, self.optimal_vmin.size)])
        vmin_guess = np.append(vmin_guess_left, vmin_guess_right)
        logeta_guess = self.optimal_logeta
        logeta_guess_left = np.maximum(logeta_guess[:vminStar_index],
                                       np.ones(vminStar_index)*logetaStar)
        logeta_guess_right = np.minimum(logeta_guess[vminStar_index:],
                                        np.ones(logeta_guess.size - vminStar_index) *
                                        logetaStar)
        logeta_guess = np.append(logeta_guess_left, logeta_guess_right)
        vars_guess = np.append(vmin_guess, logeta_guess)

        constr_func = ConstraintsFunction(vminStar, logetaStar, vminStar_index)
        constr = ({'type': 'ineq', 'fun': constr_func})
        args = (constr_func, vminStar, logetaStar, vminStar_index)

        sol_not_found = True
        attempts = 3
        np.random.seed(1)
        random_variation = 1e-5

        if USE_BASINHOPPING:
            class TakeStep(object):
                def __init__(self, stepsize=0.1):
                    pass
                    self.stepsize = stepsize

                def __call__(self, x):
                    x[:x.size/2] += np.random.uniform(-5. * self.stepsize,
                                                      5. * self.stepsize,
                                                      x[x.size/2:].shape)
                    x[x.size/2:] += np.random.uniform(-self.stepsize,
                                                      self.stepsize, x[x.size/2:].shape)
                    return x
            take_step = TakeStep()

            class AdaptiveKwargs(object):
                def __init__(self, kwargs, random_variation=random_variation):
                    self.kwargs = kwargs
                    self.random_variation = random_variation

                def __call__(self):
                    new_kwargs = {}
                    random_factor_vminStar = \
                        (1 + self.random_variation * np.random.uniform(-1, 1))
                    random_factor_logetaStar = \
                        (1 + self.random_variation * np.random.uniform(-1, 1))
                    constr_func_args = (self.kwargs['args'][1] * random_factor_vminStar,
                                        self.kwargs['args'][2] * random_factor_logetaStar,
                                        self.kwargs['args'][3])
                    constr_func = ConstraintsFunction(*constr_func_args)
                    new_kwargs['args'] = (constr_func,) + constr_func_args
                    new_kwargs['constraints'] = ({'type': 'ineq', 'fun': constr_func})
                    if 'method' in self.kwargs:
                        new_kwargs['method'] = self.kwargs['method']
                    return new_kwargs

            minimizer_kwargs = {"constraints": constr, "args": args, "method": self.method}
            if ADAPT_KWARGS:
                adapt_kwargs = AdaptiveKwargs(minimizer_kwargs, random_variation)
            else:
                adapt_kwargs = None

        while sol_not_found and attempts > 0:
            try:
                if USE_BASINHOPPING:
                    constr_optimum_log_likelihood = \
                        basinhopping(self.MinusLogLikelihood, vars_guess,
                                     minimizer_kwargs=minimizer_kwargs, niter=5,
                                     take_step=take_step, adapt_kwargs=adapt_kwargs,
                                     stepsize=0.1)
                else:
                    constr_optimum_log_likelihood = \
                        minimize(self.MinusLogLikelihood, vars_guess,
                                 args=args, constraints=constr, method=self.method)
                constraints = constr_func(constr_optimum_log_likelihood.x)
                is_not_close = np.logical_not(np.isclose(constraints,
                                                         np.zeros_like(constraints)))
                constr_not_valid = np.logical_and(constraints < 0, is_not_close)
                sol_not_found = np.any(constr_not_valid)
            except ValueError:
                sol_not_found = True
                pass

            attempts -= 1
            args = (constr_func,
                    vminStar * (1 + random_variation * np.random.uniform(-1, 1)),
                    logetaStar * (1 + random_variation * np.random.uniform(-1, 1)),
                    vminStar_index)
            if USE_BASINHOPPING:
                minimizer_kwargs = {"constraints": constr, "args": args}

            if DEBUG and sol_not_found:
                print(attempts, "attempts left! ####################################" +
                      "################################################################")
                print("sol_not_found =", sol_not_found)
        if sol_not_found:
            if DEBUG:
                print("ValueError: sol not found")
            raise ValueError

        if DEBUG:
            print(constr_optimum_log_likelihood)
            print("kwargs =", constr_optimum_log_likelihood.minimizer.kwargs)
            print("args =", constr_optimum_log_likelihood.minimizer.kwargs['args'])
            print("optimum_logL =", self.optimal_logL)
            print("constraints=", repr(constraints))
            print("constr_not_valid =", repr(constr_not_valid))
            print("vars_guess =", repr(vars_guess))
            print("optimum_logL =", self.optimal_logL)
            print("vminStar_index =", vminStar_index)

        return constr_optimum_log_likelihood

    def ConstrainedOptimalLikelihood(self, vminStar, logetaStar, plot=False):
        """ Finds the constrained minimum MinusLogLikelihood for given vminStar,
        logetaStar. Finds the minimum for all vminStar_index, and picks the best one.
        Input:
            vminStar, logetaStar: float
                Location of constrained step.
            plot: bool, optional
                Whether to plot the constrained piecewice-constant logeta function.
        Returns:
            constr_optimal_logl: float
                The constrained minimum MinusLogLikelihood
        """
        vminStar_index = 0
        while vminStar_index < self.optimal_vmin.size and \
                vminStar > self.optimal_vmin[vminStar_index]:
            vminStar_index += 1

        try:
            constr_optimum_log_likelihood = \
                self._ConstrainedOptimalLikelihood(vminStar, logetaStar, vminStar_index)
        except ValueError:
            optim_logL = 10**6
            pass
        else:
            optim_logL = constr_optimum_log_likelihood.fun
            original_optimum = constr_optimum_log_likelihood

        vminStar_index_original = vminStar_index
        index = vminStar_index
        while ALLOW_MOVE and index > 0:
            try:
                index -= 1
                new_optimum = \
                    self._ConstrainedOptimalLikelihood(vminStar, logetaStar, index)
            except ValueError:
                pass
            else:
                if new_optimum.fun < optim_logL:
                    print("Moved left, index is now", index)
                    print("############################################################" +
                          "############################################################")
                    vminStar_index = index
                    constr_optimum_log_likelihood = new_optimum
                    optim_logL = constr_optimum_log_likelihood.fun
        index = vminStar_index_original
        while ALLOW_MOVE and index < self.optimal_vmin.size:
            try:
                index += 1
                new_optimum = self._ConstrainedOptimalLikelihood(vminStar, logetaStar,
                                                                 index)
            except ValueError:
                pass
            else:
                if new_optimum.fun < optim_logL:
                    print("Moved right, index is now", index)
                    print("############################################################" +
                          "############################################################")
                    vminStar_index = index
                    constr_optimum_log_likelihood = new_optimum
                    optim_logL = constr_optimum_log_likelihood.fun
        if optim_logL == 10**6:
            raise ValueError

        self.constr_optimal_logl = constr_optimum_log_likelihood.fun
        vars_result = constr_optimum_log_likelihood.x

        self.constr_optimal_vmin = vars_result[: vars_result.size/2]
        self.constr_optimal_logeta = vars_result[vars_result.size/2:]

        if plot:
            print("vminStar =", vminStar)
            print("logetaStar =", logetaStar)
            print("vminStar_index =", vminStar_index)
            try:
                print("original:", original_optimum)
            except:
                print("Original failed.")
                pass
            try:
                print("new:", constr_optimum_log_likelihood)
                print(constr_optimum_log_likelihood.minimizer.kwargs['args'])
            except:
                print("All attepts failed.")
                pass
            try:
                vminStar_rand = constr_optimum_log_likelihood.minimizer.kwargs['args'][1]
                logetaStar_rand = constr_optimum_log_likelihood.minimizer.kwargs['args'][2]
                constr_func = ConstraintsFunction(vminStar_rand, logetaStar_rand,
                                                  vminStar_index)
                constraints = constr_func(constr_optimum_log_likelihood.x)
                is_not_close = np.logical_not(np.isclose(constraints,
                                                         np.zeros_like(constraints)))
                constr_not_valid = np.logical_and(constraints < 0, is_not_close)
                sol_not_found = np.any(constr_not_valid)
                print("random vminStar =", vminStar_rand)
                print("random logetaStar =", logetaStar_rand)
                print("x =", constr_optimum_log_likelihood.x)
                print("constraints =", constraints)
                print("is_not_close =", is_not_close)
                print("constr_not_valid =", constr_not_valid)
                print("sol_not_found =", sol_not_found)
            except:
                print("Error")
                pass

        return self.constr_optimal_logl

    def _Constrained_MC_Likelihood(self, events, vminStar, logetaStar, vminStar_index,
                                   multiexper_input, class_name,
                                   mx, fp, fn, delta):
        """ Finds the constrained minimum MinusLogLikelihood for given vminStar,
        logetaStar and vminStar_index.
        Input:
            vminStar, logetaStar: float
                Location of the constrained step.
            vminStar_index: int
                Index of vminStar in the list of vmin steps of the constrained optimum
                logeta function.
        Returns:
            constr_optimal_logl: float
                The constrained minimum MinusLogLikelihood
        """

        vmin_guess_left = np.array([events[ind]
                                    if events[ind] < vminStar
                                    else vminStar * (1 - 0.001*(vminStar_index - ind))
                                    for ind in range(vminStar_index)])
        vmin_guess_right = np.array([events[ind]
                                    if events[ind] > vminStar
                                    else vminStar * (1 + 0.001*(ind - vminStar_index + 1))
                                    for ind in range(vminStar_index, events.size)])

        vmin_guess = np.append(vmin_guess_left, vmin_guess_right)
        logeta_guess = np.array([-26.] * len(events))
        logeta_guess_left = np.maximum(logeta_guess[:vminStar_index],
                                       np.ones(vminStar_index)*logetaStar)
        logeta_guess_right = np.minimum(logeta_guess[vminStar_index:],
                                        np.ones(logeta_guess.size - vminStar_index) *
                                        logetaStar)
        logeta_guess = np.append(logeta_guess_left, logeta_guess_right)
        vars_guess = np.append(vmin_guess, logeta_guess)

        if not self.Poisson:
            constr_optimum_log_likelihood = \
                Custom_SelfConsistent_Minimization(class_name, vars_guess, mx, fp, fn, delta,
                                                   vminStar, logetaStar, vminStar_index,
                                                   vmin_err=10.0, logeta_err=0.05)
        else:
            constr = ({'type': 'ineq', 'fun': ConstraintsFunction(vminStar, logetaStar, vminStar_index)})
            logeta_bnd = (-40.0, -12.0)
            bnd_eta = [logeta_bnd] * int(vars_guess.size / 2)
            vmin_bnd = (self.minVmin, 1000.)
            bnd_vmin = [vmin_bnd] * int(vars_guess.size / 2)
            bnd = bnd_vmin + bnd_eta

            opt = minimize(self.poisson_wrapper, vars_guess,
                           args=(class_name, mx, fp, fn, delta, vminStar, logetaStar, vminStar_index),
                           jac=self.pois_jac,
                           method='SLSQP', bounds=bnd, constraints=constr, tol=1e-7,
                           options={'maxiter': 100, 'disp': False})

            if not opt.success:
                opt.fun = 1e5
            if (vminStar < self.minVmin) and (logetaStar > self.optimal_logeta[0]):
                opt.fun = self.optimal_logL
            constr_optimum_log_likelihood = [opt.x, opt.fun]

        vars_list = constr_optimum_log_likelihood[0]
        if (np.any(np.ones(int(vars_list.size/2) - 1) * (-0.01) > np.diff(vars_list[:int(vars_list.size/2)])) or
            np.any(np.ones(int(vars_list.size/2) - 1) * (-0.01) > np.diff(abs(vars_list[int(vars_list.size/2):])))):

            print('Fail ', vars_list)
            return vars_list, 10**6

        vars_list = np.insert(vars_list, vminStar_index, vminStar)
        vars_list = np.insert(vars_list, int(vars_list.size/2) + vminStar_index + 1, logetaStar)
        print(vars_list)
        return constr_optimum_log_likelihood

    def Constrained_MC_Likelihood(self, events, vminStar, logetaStar,
                                  multiexper_input, class_name,
                                  mx, fp, fn, delta):
        """ Finds the constrained minimum MinusLogLikelihood for given vminStar,
        logetaStar. Finds the minimum for all vminStar_index, and picks the best one.
        Input:
            vminStar, logetaStar: float
                Location of constrained step.
            plot: bool, optional
                Whether to plot the constrained piecewice-constant logeta function.
        Returns:
            constr_optimal_logl: float
                The constrained minimum MinusLogLikelihood
        """
        if isinstance(multiexper_input, str):
            class_name = [class_name]
        vminStar_index = 0

        for vminStar_index in range(0, len(events)+1):
            print('~~~~~~~INDEX: ', vminStar_index, '/', len(events) + 1)
            constr_optimum_new = \
                self._Constrained_MC_Likelihood(events, vminStar, logetaStar, vminStar_index,
                                                multiexper_input, class_name, mx,
                                                fp, fn, delta)
            if vminStar_index == 0:
                constr_optimum_old = constr_optimum_new[1]
            else:
                if (constr_optimum_new[1] < constr_optimum_old):
                    constr_optimum_old = constr_optimum_new[1]

        constr_optimal_logl = constr_optimum_old

        return constr_optimal_logl

    def _MultiExperConstrainedOptimalLikelihood(self, vminStar, logetaStar, vminStar_index,
                                                multiexper_input, class_name,
                                                mx, fp, fn, delta):
        """ Finds the constrained minimum MinusLogLikelihood for given vminStar,
        logetaStar and vminStar_index.
        Input:
            vminStar, logetaStar: float
                Location of the constrained step.
            vminStar_index: int
                Index of vminStar in the list of vmin steps of the constrained optimum
                logeta function.
        Returns:
            constr_optimal_logl: float
                The constrained minimum MinusLogLikelihood
        """

        if DEBUG:
            print("~~~~~ vminStar_index =", vminStar_index)
        vmin_guess_left = np.array([self.optimal_vmin[ind]
                                    if self.optimal_vmin[ind] < vminStar
                                    else vminStar * (1 - 0.001*(vminStar_index - ind))
                                    for ind in range(vminStar_index)])
        vmin_guess_right = np.array([self.optimal_vmin[ind]
                                    if self.optimal_vmin[ind] > vminStar
                                    else vminStar * (1 + 0.001*(ind - vminStar_index + 1))
                                    for ind in range(vminStar_index, self.optimal_vmin.size)])

        vmin_guess = np.append(vmin_guess_left, vmin_guess_right)
        logeta_guess = self.optimal_logeta
        logeta_guess_left = np.maximum(logeta_guess[:vminStar_index],
                                       np.ones(vminStar_index)*logetaStar)
        logeta_guess_right = np.minimum(logeta_guess[vminStar_index:],
                                        np.ones(logeta_guess.size - vminStar_index) *
                                        logetaStar)
        logeta_guess = np.append(logeta_guess_left, logeta_guess_right)
        vars_guess = np.append(vmin_guess, logeta_guess)
        if not self.Poisson:
            constr_optimum_log_likelihood = \
                Custom_SelfConsistent_Minimization(class_name, vars_guess, mx, fp, fn, delta,
                                                   vminStar, logetaStar, vminStar_index,
                                                   vmin_err=9.0, logeta_err=0.02)
        else:
            constr = ({'type': 'ineq', 'fun': ConstraintsFunction(vminStar, logetaStar, vminStar_index)})
            logeta_bnd = (-40.0, -12.0)
            bnd_eta = [logeta_bnd] * int(vars_guess.size / 2)
            vmin_bnd = (0., 1000.)
            bnd_vmin = [vmin_bnd] * int(vars_guess.size / 2)
            bnd = bnd_vmin + bnd_eta

            opt = minimize(self.poisson_wrapper, vars_guess,
                           args=(class_name, mx, fp, fn, delta,
                                 vminStar, logetaStar, vminStar_index),
                           jac=self.pois_jac,
                           method='SLSQP', bounds=bnd, constraints=constr, tol=1e-4,
                           options={'maxiter': 100, 'disp': False})

            constr_optimum_log_likelihood = [opt.x, opt.fun]

        if DEBUG:
            print(constr_optimum_log_likelihood[0])
            print("Constrained optimum_logL =", constr_optimum_log_likelihood[1])
            print("vminStar_index =", vminStar_index)

        return constr_optimum_log_likelihood

    def MultiExperConstrainedOptimalLikelihood(self, vminStar, logetaStar,
                                               multiexper_input, class_name,
                                               mx, fp, fn, delta, plot=False):
        """ Finds the constrained minimum MinusLogLikelihood for given vminStar,
        logetaStar. Finds the minimum for all vminStar_index, and picks the best one.
        Input:
            vminStar, logetaStar: float
                Location of constrained step.
            plot: bool, optional
                Whether to plot the constrained piecewice-constant logeta function.
        Returns:
            constr_optimal_logl: float
                The constrained minimum MinusLogLikelihood
        """

        events = self.optimal_vmin
        for vminStar_index in range(0, len(events)+1):
            print('~~~~~~~INDEX: ', vminStar_index, '/', len(events) + 1)
            vars_result, constr_optimum_new = \
                self._Constrained_MC_Likelihood(events, vminStar, logetaStar, vminStar_index,
                                                multiexper_input, class_name, mx,
                                                fp, fn, delta)
            print('constrained optimum: ', constr_optimum_new)
            if vminStar_index == 0:
                constr_optimum_old = constr_optimum_new
                vars_result_old = vars_result
            else:
                if constr_optimum_new < constr_optimum_old:
                    constr_optimum_old = constr_optimum_new
                    vars_result_old = vars_result

        constr_optimal_logl = constr_optimum_old
        self.constr_optimal_logl = constr_optimal_logl

        self.constr_optimal_vmin = vars_result[: int(vars_result.size/2)]
        self.constr_optimal_logeta = vars_result[int(vars_result.size/2):]

        print("vminStar =", vminStar)
        print("logetaStar =", logetaStar)
        print("new:", self.constr_optimal_logl, vars_result_old)

        return self.constr_optimal_logl

    def VminSamplingList(self, output_file_tail, output_file_CDMS, vmin_min, vmin_max, vmin_num_steps,
                         steepness_vmin=1.5, steepness_vmin_center=2.5, MULTI_EXPER=False,
                         plot=False):
        """ Finds a non-linear way to sample the vmin range, such that more points are
        sampled near the location of the steps of the best-fit logeta function, and
        fewer in between. This is done by building a function of vmin that is steeper
        near the steps and flatter elsewhere, and the steeper this function the more
        samplings are done in this region.
        Input:
            output_file_tail: string
                Tag to be added to the file name.
            vmin_min, vmin_max: float
                Range in vmin where the sampling should be made.
            vmin_num_steps: int
                Number of samples in vmin (approximate, the final number of steps is
                not exact, due to taking floor() in some places.
            steepness_vmin: float, optional
                Parameter related to the steepness of this function to the left of the
                leftmost step and to the right of the rightmost step.
            steepness_vmin_center: float, optional
                Similar parameter, but for the steepness in between the leftmost step
                and the rightmost step.
            plot: bool, optional
                Whether to plot intermediate results such as the sampling function.
        """
        if not MULTI_EXPER:
            self.ImportOptimalLikelihood(output_file_tail)
        else:
            self.ImportMultiOptimalLikelihood(output_file_tail, output_file_CDMS)
        xmin = vmin_min
        xmax = vmin_max
        # TODO! This +4 is to compensate for a loss of ~4 points (not always 4 though),
        # and it's due to taking floor later on.
        # Find a better way to deal with this.
        x_num_steps = vmin_num_steps  # + 4
        s = steepness_vmin
        sc = steepness_vmin_center

        x_lin = np.linspace(xmin, xmax, 1000)
        x0_list = self.optimal_vmin
        numx0 = x0_list.size

        print("x0 =", x0_list)

        def UnitStep(x): return (np.sign(x) + 1) / 2

        def g1(x, x0, s0, xmin=xmin):
            return np.log10(UnitStep(x - x0) +
                            UnitStep(x0 - x) *
                            (x0 - xmin) / (x + 10**s0 * (-x + x0) - xmin))

        def g2(x, x0, s0, xmax=xmax):
            return np.log10(UnitStep(x0 - x) +
                            UnitStep(x - x0) *
                            (x + 10**s0 * (-x + x0) - xmax) / (x0 - xmax))

        def g(x, x0, s1, s2): return g1(x, x0, s1) + g2(x, x0, s2)

        s_list = np.array([[s, sc]] + [[sc, sc]] * (numx0 - 2) + [[sc, s]])

        def g_total(x, sign=1, x0=x0_list, s_list=s_list):
            return np.array([sign * g(x, x0_list[i], s_list[i, 0], s_list[i, 1])
                             for i in range(x0_list.size)]).prod(axis=0)

        g_lin = g_total(x_lin)

        xT_guess = (x0_list[:-1] + x0_list[1:]) / 2
        bounds = np.array([(x0_list[i], x0_list[i + 1])
                           for i in range(x0_list.size - 1)])
        x_turns_max = np.array([minimize(g_total, np.array(xT_guess[i]),
                                args=(-1,), bounds=[bounds[i]]).x
                                for i in range(0, xT_guess.size, 2)])
        x_turns_min = np.array([minimize(g_total, np.array(xT_guess[i]),
                                         bounds=[bounds[i]]).x
                                for i in range(1, xT_guess.size, 2)])
        x_turns = np.sort(np.append(x_turns_max, x_turns_min))
        x_turns = np.append(np.insert(x_turns, 0, xmin), [xmax])
        y_turns = g_total(x_turns)
        print("x_turns =", x_turns)
        print("y_turns =", y_turns)

        def g_inverse(y, x1, x2):
            return brentq(lambda x: g_total(x) - y, x1, x2)

        def g_inverse_list(y_list, x1, x2):
            return np.array([g_inverse(y, x1, x2) for y in y_list])

        y_diff = np.diff(y_turns)
        y_diff_sum = np.abs(y_diff).sum()
        print("y_diff =", y_diff)
        num_steps = np.array([max(1, np.floor(x_num_steps * np.abs(yd)/y_diff_sum))
                              for yd in y_diff])
        print("num_steps =", num_steps)
        y_list = np.array([np.linspace(y_turns[i], y_turns[i+1], num_steps[i])
                           for i in range(num_steps.size)])

        try:
            x_list = np.array([g_inverse_list(y_list[i], x_turns[i], x_turns[i+1])
                               for i in range(y_list.size)])
        except IndexError:
            x_list = np.array([g_inverse_list(y_list[i], x_turns[i], x_turns[i + 1])
                               for i in range(y_list[:,0].size)])
        x_list = np.concatenate(x_list)
        y_list = np.concatenate(y_list)
        x_list = x_list[np.array([x_list[i] != x_list[i+1]
                        for i in range(x_list.size - 1)] + [True])]
        y_list = y_list[np.array([y_list[i] != y_list[i+1]
                        for i in range(y_list.size - 1)] + [True])]
        self.vmin_sampling_list = x_list

        if plot:
            plt.close()
            plt.plot(x_lin, g_lin)
            plt.plot(x_turns, y_turns, 'o')
            plt.plot(x_list, y_list, '*')
            plt.xlim([xmin, xmax])
            plt.ylim([min(-s * sc**(numx0 - 1), np.min(y_turns)),
                      max(s * sc**(numx0 - 1), np.max(y_turns))])
            plt.show()

        return

    def OptimumStepFunction(self, vmin):
        """ Best-fit logeta as a function of vmin for the optimal log(L).
        Input:
            vmin: float
                Value of vmin for which to evaluate logeta.
        Returns:
            logeta: float
                log(eta(vmin)) for the best-fit piecewise constant function.
        """
        index = 0
        while index < self.optimal_vmin.size and vmin > self.optimal_vmin[index]:
            index += 1
        if index == self.optimal_vmin.size:
            return self.optimal_logeta[-1]*10
        return self.optimal_logeta[index]

    def VminLogetaSamplingTable(self, output_file_tail, logeta_percent_minus,
                                logeta_percent_plus, logeta_num_steps,
                                linear_sampling=True, steepness_logeta=1, plot=False):
        """ Finds a non-linear way to sample both the vmin and logeta range, such that
        more points are sampled near the location of the steps of the best-fit logeta
        function, and fewer in between. This uses the sampling in vmin done by
        VminSamplingList, and computes a non-linear sampling in logeta in a similar way
        (by building a function of logeta that is steeper near the steps and flatter
        elsewhere, and the steeper this function the more samplings are done in this
        region).
        Input:
            output_file_tail: string
                Tag to be added to the file name.
            logeta_percent_minus, logeta_percent_plus: float
                Range in logeta where the sampling should be made, given as percentage
                in the negative and positive direction of the best-fit logeta.
            logeta_num_steps: int
                Number of samples in logeta.
            steepness_logeta: float, optional
                Parameter related to the steepness of this sampling function in logeta.
            plot: bool, optional
                Whether to plot intermediate results such as the sampling function.
        """
        print(self.optimal_vmin)
        print(self.optimal_logeta)

        logeta_num_steps_minus = logeta_num_steps * \
            logeta_percent_minus / (logeta_percent_minus + logeta_percent_plus)
        logeta_num_steps_plus = logeta_num_steps * \
            logeta_percent_plus / (logeta_percent_minus + logeta_percent_plus)

        s = steepness_logeta

        def f(x, xm, i, s0=s):
            return (xm - x) / (10**s0 - 1) * 10**i + (10**s0 * x - xm) / (10**s0 - 1)

        self.vmin_logeta_sampling_table = []
        vmin_last_step = self.optimal_vmin[-1]

        if linear_sampling:
            for vmin in self.vmin_sampling_list:
                logeta_opt = self.OptimumStepFunction(min(vmin, vmin_last_step))
                if vmin < self.optimal_vmin[0]:
                    logeta_min = logeta_opt * (1 + 0.6 * logeta_percent_minus)
                    logeta_max = logeta_opt * (1 - logeta_percent_plus)
                else:
                    if vmin < 600:
                        logeta_min = logeta_opt * (1 + logeta_percent_minus)
                    else:
                        logeta_min = logeta_opt * (1 + 0.6 * logeta_percent_minus)
                    logeta_max = logeta_opt * (1 - 0.5 * logeta_percent_plus)
                logeta_list = [[vmin, logeta]
                               for logeta in np.linspace(logeta_min, logeta_max,
                                                         logeta_num_steps)]
                self.vmin_logeta_sampling_table += [logeta_list]
        else:
            for vmin in self.vmin_sampling_list:
                logeta_opt = self.OptimumStepFunction(min(vmin, vmin_last_step))
                logeta_min = logeta_opt * (1 + logeta_percent_minus)
                logeta_max = logeta_opt * (1 - logeta_percent_plus)
                logeta_list_minus = [[vmin, f(logeta_opt, logeta_min, i)]
                                     for i in np.linspace(s, 0, logeta_num_steps_minus)]
                logeta_list_plus = [[vmin, f(logeta_opt, logeta_max, i)]
                                    for i in np.linspace(s / logeta_num_steps_plus, s,
                                                         logeta_num_steps_plus)]
                self.vmin_logeta_sampling_table += [logeta_list_minus + logeta_list_plus]

        self.vmin_logeta_sampling_table = np.array(self.vmin_logeta_sampling_table)

        if plot:
            self.PlotSamplingTable(plot_close=True)
        return

    def PlotSamplingTable(self, plot_close=False, plot_show=True, plot_optimum=True):
        """ Plots the sampling points in the vmin-logeta plane.
        """
        if plot_close:
            plt.close()
        print("sampling_size =", self.vmin_logeta_sampling_table.shape)
        for tab in self.vmin_logeta_sampling_table:
            plt.plot(tab[:, 0], tab[:, 1], 'o')
        if plot_optimum:
            self.PlotOptimum(xlim_percentage=(0.9, 1.1), ylim_percentage=(1.2, 0.8),
                             plot_close=False, plot_show=plot_show)
        elif plot_show:
            plt.show()
        return

    def GetLikelihoodTable(self, index, output_file_tail, logeta_index_range, extra_tail):
        """ Prints to file lists of the form [logetaStar_ij, logL_ij] needed for
        1D interpolation, where i is the index corresponding to vminStar_i and j is
        the index for each logetaStar. Each file corresponds to a different index i.
            Here only one file is written for a specific vminStar.
        Input:
            index: int
                Index of vminStar.
            output_file_tail: string
                Tag to be added to the file name.
            logeta_index_range: tuple
                A tuple (index0, index1) between which logetaStar will be considered.
                If this is None, then the whole list of logetaStar is used.
            extra_tail: string
                Additional tail to be added to filenames.
        """
        print('index =', index)
        print('output_file_tail =', output_file_tail)
        vminStar = self.vmin_logeta_sampling_table[index, 0, 0]
        logetaStar_list = self.vmin_logeta_sampling_table[index, :, 1]
        plot = False
        if logeta_index_range is not None:
            logetaStar_list = \
                logetaStar_list[logeta_index_range[0]: logeta_index_range[1]]
            plot = True

        print("vminStar =", vminStar)
        table = np.empty((0, 2))
        for logetaStar in logetaStar_list:
            try:
                constr_opt = self.ConstrainedOptimalLikelihood(vminStar, logetaStar,
                                                               plot=plot)
            except:
                print("error")
                os.system("say Error")
                pass
            else:
                print("index =", index, "; vminStar =", vminStar,
                      "; logetaStar =", logetaStar, "; constr_opt =", constr_opt)
                table = np.append(table, [[logetaStar, constr_opt]], axis=0)
#                table = np.append(table, [logetaStar])
        print("vminStar =", vminStar, "; table =", table)
        if True:
            temp_file = output_file_tail + "_" + str(index) + \
                "_LogetaStarLogLikelihoodList" + extra_tail + ".dat"
            print(temp_file)
            np.savetxt(temp_file, table)
        return

    def GetLikelihoodTableMultiExper(self, index, output_file_tail, logeta_index_range, extra_tail,
                                     multiexper_input,
                                     mx, fp, fn, delta, scattering_type,
                                     mPhi, quenching):
        """ Prints to file lists of the form [logetaStar_ij, logL_ij] needed for
        1D interpolation, where i is the index corresponding to vminStar_i and j is
        the index for each logetaStar. Each file corresponds to a different index i.
            Here only one file is written for a specific vminStar.
        Input:
            index: int
                Index of vminStar.
            output_file_tail: string
                Tag to be added to the file name.
            logeta_index_range: tuple
                A touple (index0, index1) between which logetaStar will be considered.
                If this is None, then the whole list of logetaStar is used.
            extra_tail: string
                Additional tail to be added to filenames.
        """

        print('index =', index)
        print('output_file_tail =', output_file_tail)

        vminStar = self.vmin_logeta_sampling_table[index, 0, 0]
        logetaStar_list = self.vmin_logeta_sampling_table[index, :, 1]
        plot = False

        if logeta_index_range is not None:
            logetaStar_list = \
                logetaStar_list[logeta_index_range[0]: logeta_index_range[1]]
            plot = False
        # print("vminStar =", vminStar)
        table = np.empty((0, 2))

        temp_file = output_file_tail + "_" + str(index) + \
            "_LogetaStarLogLikelihoodList" + extra_tail + ".dat"
        print('TempFile: ', temp_file)
        if os.path.exists(temp_file):
            size_of_file = len(np.loadtxt(temp_file))
            fileexists = True
        else:
            print('File doesnt exist')
            size_of_file = 0
            fileexists = False
        if size_of_file >= 30:
            pass
        else:
            if fileexists and size_of_file > 1 and np.loadtxt(temp_file).ndim == 2:
                table = np.loadtxt(temp_file)
                for logetaStar in logetaStar_list:
                    if logetaStar > table[-1, 0]:
                        print('V* =', vminStar, 'log(eta)* =', logetaStar)
                        constr_opt = self.MultiExperConstrainedOptimalLikelihood(vminStar, logetaStar,
                                                                                 multiexper_input,
                                                                                 self.class_name, mx,
                                                                                 fp, fn, delta, plot)

                        print("index =", index, "; vminStar =", vminStar,
                              "; logetaStar =", logetaStar, "; constr_opt =", constr_opt)
                        table = np.append(table, [[logetaStar, constr_opt]], axis=0)

                        print("vminStar =", vminStar, "; table =", table)

                        print(temp_file)
                        np.savetxt(temp_file, table)
            else:
                for logetaStar in logetaStar_list:
                    print('V* =', vminStar, 'log(eta)* =', logetaStar)
                    constr_opt = self.MultiExperConstrainedOptimalLikelihood(vminStar, logetaStar,
                                                                             multiexper_input,
                                                                             self.class_name, mx,
                                                                             fp, fn, delta, plot)

                    print("index =", index, "; vminStar =", vminStar,
                          "; logetaStar =", logetaStar, "; constr_opt =", constr_opt)
                    table = np.append(table, [[logetaStar, constr_opt]], axis=0)

                    print("vminStar =", vminStar, "; table =", table)

                    print(temp_file)
                    print('Saved Line.')
                    np.savetxt(temp_file, table)
        return

    def LogLikelihoodList(self, output_file_tail, extra_tail="", processes=None,
                          vmin_index_list=None, logeta_index_range=None):
        """ Loops thorugh the list of all vminStar and calls GetLikelihoodTable,
        which will print the likelihood tables to files.
        Input:
            output_file_tail: string
                Tag to be added to the file name.
            extra_tail: string, optional
                Additional tail to be added to filenames.
            processes: int, optional
                Number of processes for parallel programming.
            vmin_index_list: ndarray, optional
                List of indices in vminStar_list for which we calculate the optimal
                likelihood. If not given, the whole list of vminStars is used.
            logeta_index_range: tuple, optional
                Atuple (index0, index1) between which logetaStar will be considered.
                If not given, then the whole list of logetaStar is used.
        """
        if vmin_index_list is None:
            vmin_index_list = range(0, self.vmin_logeta_sampling_table.shape[0])
        else:
            try:
                len(vmin_index_list)
            except TypeError:
                vmin_index_list = range(vmin_index_list,
                                        self.vmin_logeta_sampling_table.shape[0])
        print("vmin_index_list =", vmin_index_list)
        print("logeta_index_range =", logeta_index_range)
        kwargs = ({'index': index,
                   'output_file_tail': output_file_tail,
                   'logeta_index_range': logeta_index_range,
                   'extra_tail': extra_tail}
                  for index in vmin_index_list)
        par.parmap(self.GetLikelihoodTable, kwargs, processes)

        return

    def MultiExperLogLikelihoodList(self, output_file_tail, multiexper_input, class_name,
                                    mx, fp, fn, delta, scattering_type, mPhi, quenching,
                                    extra_tail="", processes=None,
                                    vmin_index_list=None, logeta_index_range=None):
        """ Loops thorugh the list of all vminStar and calls GetLikelihoodTable,
        which will print the likelihood tables to files.
        Input:
            output_file_tail: string
                Tag to be added to the file name.
            extra_tail: string, optional
                Additional tail to be added to filenames.
            processes: int, optional
                Number of processes for parallel programming.
            vmin_index_list: ndarray, optional
                List of indices in vminStar_list for which we calculate the optimal
                likelihood. If not given, the whole list of vminStars is used.
            logeta_index_range: tuple, optional
                Atuple (index0, index1) between which logetaStar will be considered.
                If not given, then the whole list of logetaStar is used.
        """
        if vmin_index_list is None:
            vmin_index_list = range(0, self.vmin_logeta_sampling_table.shape[0])
        else:
            try:
                len(vmin_index_list)
            except TypeError:
                vmin_index_list = range(vmin_index_list,
                                        self.vmin_logeta_sampling_table.shape[0])
        self.class_name = class_name
        print("vmin_index_list =", vmin_index_list)
        print("logeta_index_range =", logeta_index_range)
        kwargs = ({'index': index,
                   'output_file_tail': output_file_tail,
                   'logeta_index_range': logeta_index_range,
                   'extra_tail': extra_tail,
                   'multiexper_input': multiexper_input,
                   'scattering_type': scattering_type,
                   'mPhi': mPhi,
                   'quenching': quenching,
                   'mx': mx,
                   'fp': fp,
                   'fn': fn,
                   'delta': delta}
                  for index in vmin_index_list)

        par.parmap(self.GetLikelihoodTableMultiExper, kwargs, processes)
#        for y in vmin_index_list:
#            self.GetLikelihoodTableMultiExper(y, output_file_tail, logeta_index_range,
#                            extra_tail, multiexper_input, class_name, mx, fp, fn, delta,
#                            scattering_type, mPhi, quenching,)
        return

    def _logL_interp(vars_list, constraints):
        constr_not_valid = constraints(vars_list)[:-1] < 0
        if np.any(constr_not_valid):
            constr_list = constraints(vars_list)[constr_not_valid]
            return -constr_list.sum() * 10**2
        return logL_interp(vars_list)

    def ConfidenceBand(self, output_file_tail, delta_logL, interpolation_order,
                       extra_tail="", multiplot=True):
        """ Compute the confidence band.
        Input:
            output_file_tail: string
                Tag to be added to the file name.
            delta_logL: float
                Target difference between the constrained minimum and the
                unconstrained global minimum of MinusLogLikelihood.
            interpolation_order: int
                interpolation order for the interpolated constrained minimum of
                MinusLogLikelihood as a function of logeta, for a fixed vmin.
            extra_tail: string, optional
                Additional tail to be added to filenames.
            multiplot: bool, optional
                Whether to plot log(L) as a function of logeta for each vmin, and the
                horizontal line corresponding to a given delta_logL.
        """
        print("self.vmin_sampling_list =", self.vmin_sampling_list)
        self.vmin_logeta_band_low = []
        self.vmin_logeta_band_up = []
        vmin_last_step = self.optimal_vmin[-1]
        if multiplot:
            plt.close()
        for index in range(self.vmin_sampling_list.size):
            print("index =", index)
            print("vmin =", self.vmin_sampling_list[index])
            logeta_optim = self.OptimumStepFunction(min(self.vmin_sampling_list[index],
                                                        vmin_last_step))
            file = output_file_tail + "_" + str(index) + \
                "_LogetaStarLogLikelihoodList" + extra_tail + ".dat"
            try:
                with open(file, 'r') as f_handle:
                    table = np.loadtxt(f_handle)
            except:
                continue
            x = table[:, 0]   # this is logeta
            y = table[:, 1]   # this is logL
            logL_interp = interpolate.interp1d(x, y, kind='cubic')

            def _logL_interp(vars_list, constraints):
                constr_not_valid = constraints(vars_list)[:-1] < 0
                if np.any(constr_not_valid):
                    constr_list = constraints(vars_list)[constr_not_valid]
                    return -constr_list.sum() * 1e2
                return logL_interp(vars_list)

            print(self.optimal_logL - delta_logL)
            print(np.array([table[0, 0]]), " ", table[-1, 0])
            print(logeta_optim)

            def constr_func(logeta, logeta_min=np.array([table[0, 0]]),
                            logeta_max=np.array([table[-1, 0]])):
                return np.concatenate([logeta - logeta_min, logeta_max - logeta])

            constr = ({'type': 'ineq', 'fun': constr_func})
            try:
                logeta_minimLogL = minimize(_logL_interp, np.array([logeta_optim]),
                                            args=(constr_func,), constraints=constr).x[0]
            except ValueError:
                print("ValueError at logeta_minimLogL")
                logeta_minimLogL = logeta_optim
                pass
            print("logeta_minimLogL =", logeta_minimLogL)

            print("x =", x)
            print("y =", y)
            if multiplot:
                plt.close()
                plt.plot(x, y, 'o-')
                plt.plot(x, (self.optimal_logL + 1) * np.ones_like(y))
                plt.plot(x, (self.optimal_logL + 2.7) * np.ones_like(y))
                plt.title("index =" + str(index) + ", v_min =" +
                          str(self.vmin_sampling_list[index]) + "km/s")
                plt.xlim(x[0], x[-1])
                plt.ylim(-5, 20)
                plt.show()

            error = F

            try:
                if y[0] > self.optimal_logL + delta_logL:  # and abs(logeta_minimLogL) < self.optimal_logL + delta_logL:
                    sol = brentq(lambda logeta: logL_interp(logeta) - self.optimal_logL -
                                 delta_logL,
                                 table[0, 0], logeta_minimLogL)

                    self.vmin_logeta_band_low += \
                        [[self.vmin_sampling_list[index], sol]]
                else:
                    self.vmin_logeta_band_low += \
                        [[self.vmin_sampling_list[index], -40.]]
            except ValueError:
                print("ValueError: Error in calculating vmin_logeta_band_low")
                error = T

            try:
                if y[-1] > self.optimal_logL + delta_logL and \
                        logeta_minimLogL < self.optimal_logL + delta_logL:
                    sol = brentq(lambda logeta: logL_interp(logeta) - self.optimal_logL -
                                 delta_logL,
                                 logeta_minimLogL, table[-1, 0])
                    self.vmin_logeta_band_up += \
                        [[self.vmin_sampling_list[index], sol]]

            except ValueError:
                print("ValueError: Error in calculating vmin_logeta_band_hi")
                error = T

            if error:
                plt.close()
                plt.plot(x, (self.optimal_logL + 1) * np.ones_like(y))
                plt.plot(x, (self.optimal_logL + 2.7) * np.ones_like(y))
                plt.title("index =" + str(index) + "; v_min =" +
                          str(self.vmin_sampling_list[index]) + "km/s")
                plt.xlim(x[0], x[-1])
                plt.ylim([-5, 20])
                plt.plot(x, y, 'o-', color="r")
                plt.plot(logeta_optim, logL_interp(logeta_optim), '*')
                plt.plot(logeta_optim, self.optimal_logL, '*')
                print("ValueError")
                plt.show()
#                raise
                pass

        if multiplot:
            plt.show()

        self.vmin_logeta_band_low = np.array(self.vmin_logeta_band_low)
        self.vmin_logeta_band_up = np.array(self.vmin_logeta_band_up)

        print("lower band: ", self.vmin_logeta_band_low)
        print("upper band: ", self.vmin_logeta_band_up)

        self.PlotConfidenceBand()

        delta_logL = round(delta_logL, 1)
        file = output_file_tail + "_FoxBand_low_deltalogL_" + str(delta_logL) + ".dat"
        print(file)
        np.savetxt(file, self.vmin_logeta_band_low)
        file = output_file_tail + "_FoxBand_up_deltalogL_" + str(delta_logL) + ".dat"
        print(file)
        np.savetxt(file, self.vmin_logeta_band_up)

        return

    def PlotConfidenceBand(self):
        """ Plot the confidence band and the best-fit function.
        """
        plt.close()
        try:
            plt.plot(self.vmin_logeta_band_low[:, 0], self.vmin_logeta_band_low[:, 1], 'o-')
        except IndexError:
            pass
        try:
            plt.plot(self.vmin_logeta_band_up[:, 0], self.vmin_logeta_band_up[:, 1], 'o-')
        except IndexError:
            pass
        self.PlotOptimum(ylim_percentage=(1.2, 0.8), plot_close=F, plot_show=T)

    def ImportConfidenceBand(self, output_file_tail, delta_logL, extra_tail=""):
        """ Import the confidence band from file.
        Input:
            output_file_tail: string
                Tag to be added to the file name.
            delta_logL: float
                Target difference between the constrained minimum and the
                unconstrained global minimum of MinusLogLikelihood.
            extra_tail: string, optional
                Additional tail to be added to filenames.
        """
        delta_logL = round(delta_logL, 1)
        file = output_file_tail + "_FoxBand_low_deltalogL_" + str(delta_logL) + \
            extra_tail + ".dat"
        print(file)
        with open(file, 'r') as f_handle:
            self.vmin_logeta_band_low = np.loadtxt(f_handle)
        file = output_file_tail + "_FoxBand_up_deltalogL_" + str(delta_logL) + \
            extra_tail + ".dat"
        with open(file, 'r') as f_handle:
            self.vmin_logeta_band_up = np.loadtxt(f_handle)
        return
