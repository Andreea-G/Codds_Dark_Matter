# -*- coding: utf-8 -*-
"""
Created on Fri Feb  5 13:00:53 2016

@author: SamWitte


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

from __future__ import division, print_function, absolute_import

import numpy as np
from numpy import cos, sin
from math import factorial, log
from basinhopping import *
from scipy.optimize import brentq, minimize
import collections






"""
The general idea here is this:
Can I create a minimization procedure that is more efficient than that standard basinhopping (and finds the actual minimium).
To do this, I plan on picking my value of vmin, minimizing wrt to the value of eta, then basinhopping at a fixed
eta across different vmin, then reminimize wrt to eta, etc.

Do this procedure until a self consistent minimum is found. I will start by writing this code for CDMS-Si and 1 other
poisson binned experiment, but I'd like to generalize at some point in the future.

TODO Generalize to: more than 2 experiments, different likelihood functions, larger N_o -- also clean up and write neater code
"""



#class CustomMinimize():
#    def __init__(self, class_names):
#        self.class_names = class_names

def Custom_SelfConsistent_Minimization(class_name, x0, mx, fp, fn, delta,
                                           bh_iter = 15, vmin_err = 7.0, logeta_err = 0.01,
                                           vmin_step = 15.0):

        
        vmin_list = x0[: len(x0)/2]
        logeta_list = x0[len(x0)/2 :]
        vmin_listw0 = np.insert(vmin_list, 0, 0)
        rate_partials = [None] * (class_name[1].BinEdges_left.size)
        script_N_a = class_name[0].IntegratedResponseTable(vmin_listw0)
        script_M_a = class_name[0].VminIntegratedResponseTable(vmin_listw0)


        for x in range(0, class_name[1].BinEdges_left.size):
            resp_integr = class_name[1].IntegratedResponseTable(vmin_listw0,
                      class_name[1].BinEdges_left[x], class_name[1].BinEdges_right[x], mx, fp, fn, delta)

            rate_partials[x] = resp_integr

        def constr_func_logeta(x):

            constraints = np.concatenate([-x, np.diff(-x)])

            is_not_close = np.logical_not(
                np.isclose(constraints, np.zeros_like(constraints), atol=1e-3))
            is_not_close[x.size] = True
            constr = np.where(is_not_close, constraints, np.abs(constraints))

#            print("***constr =", repr(constr))
#            print("tf =", repr(constr < 0))

            return constr

        def constr_func_vmin(x):

            constraints = np.concatenate([x, np.diff(x)])

            is_not_close = np.logical_not(
                np.isclose(constraints, np.zeros_like(constraints), atol=1e-3))
            is_not_close[x.size] = True
            constr = np.where(is_not_close, constraints, np.abs(constraints))

#            print("***constr =", repr(constr))
#            print("tf =", repr(constr < 0))

            return constr

        constr = ({'type': 'ineq', 'fun': constr_func_logeta})
        constr_vmin = ({'type': 'ineq', 'fun': constr_func_vmin})


        def optimize_logeta(logeta_list, class_name, script_N_a, script_M_a,
                    rate_partials):

            optimize_func = (2.0 * (class_name[0].NBKG + class_name[0].Exposure * np.dot(10**logeta_list, script_N_a) -
                np.log(class_name[0].mu_BKG_i + class_name[0].Exposure * np.dot(script_M_a, 10**logeta_list)).sum()))

            for x in range(0, class_name[1].BinData.size):

                if (class_name[1].Exposure * np.dot(10**logeta_list, rate_partials[x])) > class_name[1].BinData[x]:
                    optimize_func += 2.0 * (class_name[1].Exposure * np.dot(10**logeta_list, rate_partials[x]) + log(factorial(class_name[1].BinData[x])) -
                                 class_name[1].BinData[x] * log(class_name[1].Exposure * np.dot(10**logeta_list, rate_partials[x])))
                elif class_name[1].BinData[x] > (class_name[1].Exposure * np.dot(10**logeta_list, rate_partials[x])):
                        optimize_func += 2.0 * (class_name[1].BinData[x] + log(factorial(class_name[1].BinData[x])) -
                                       class_name[1].BinData[x] * log(class_name[1].BinData[x]))

            return optimize_func

        def minimize_over_vmin(vmin_list, logeta_list, class_name, mx, fp, fn, delta):

            vmin_listw0 = np.insert(vmin_list, 0, 0)
            rate_partials = [None] * (class_name[1].BinEdges_left.size)

            script_N_a = class_name[0].IntegratedResponseTable(vmin_listw0)
            script_M_a = class_name[0].VminIntegratedResponseTable(vmin_listw0)

            optimize_func = (2.0 * (class_name[0].NBKG + class_name[0].Exposure * np.dot(10**logeta_list, script_N_a) -
                np.log(class_name[0].mu_BKG_i + class_name[0].Exposure * np.dot(script_M_a, 10**logeta_list)).sum()))

            for x in range(0, class_name[1].BinEdges_left.size):
                resp_integr = class_name[1].IntegratedResponseTable(vmin_listw0,
                          class_name[1].BinEdges_left[x], class_name[1].BinEdges_right[x], mx, fp, fn, delta)
                rate_partials[x] = resp_integr

                if (class_name[1].Exposure * np.dot(10**logeta_list, rate_partials[x])) > class_name[1].BinData[x]:
                        optimize_func += 2.0 * (class_name[1].Exposure * np.dot(10**logeta_list, rate_partials[x]) + log(factorial(class_name[1].BinData[x])) -
                                 class_name[1].BinData[x] * log(class_name[1].Exposure * np.dot(10**logeta_list, rate_partials[x])))
                elif class_name[1].BinData[x] > (class_name[1].Exposure * np.dot(10**logeta_list, rate_partials[x])):
                        optimize_func += 2.0 * (class_name[1].BinData[x] + log(factorial(class_name[1].BinData[x])) -
                                       class_name[1].BinData[x] * log(class_name[1].BinData[x]))

            return optimize_func

        ni = 0
        check = False

        while ni < 15 and not check:
            mloglike_min = minimize(optimize_logeta, logeta_list, args=(class_name, script_N_a, script_M_a,
                                rate_partials), method = 'SLSQP',
                                bounds = [(-35.0, -24.0),(-35.0, -24.0),(-35.0, -24.0)], constraints=constr)
            logeta_list_new = mloglike_min.x

            minimizer_kwargs = {"method": "SLSQP", "constraints": constr_vmin,
                            "args": (logeta_list_new, class_name, mx, fp, fn, delta),
                            "options": {'ftol': 1.0}}
            vminloglike_min = basinhopping(minimize_over_vmin, vmin_list,
                        minimizer_kwargs=minimizer_kwargs, niter=bh_iter, stepsize=vmin_step)

            vmin_list_new = vminloglike_min.x

            vmin_check = np.abs(np.asarray(vmin_list_new) - np.asarray(vmin_list))
            logeta_check = np.abs(np.asarray(logeta_list_new) - np.asarray(logeta_list))

            if np.amax(vmin_check) < vmin_err and np.amax(logeta_check) < logeta_err:
                print('n_iter', ni)
                print('Minimum Found')
                check = True
            else:
                print('n_iter', ni)
                print('v_min old', vmin_list)
                print('v_min new', vmin_list_new)
                print('logeta old', logeta_list)
                print('logeta new', logeta_list_new)
                print('vmin_check', vmin_check)
                print('logeta_check', logeta_check)
                vmin_list = vmin_list_new
                logeta_list = logeta_list_new
            ni += 1

        return [np.concatenate([vmin_list_new, logeta_list_new]), vminloglike_min.fun]




