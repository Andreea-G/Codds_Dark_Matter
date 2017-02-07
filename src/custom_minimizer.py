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

import numpy as np
from scipy.optimize import minimize


"""
Pick a value of vmin, minimizing wrt to the
value of eta (this can be done very quickly,
this is the advantage over alternative basinhopping method), 
then minimizing at a fixed eta across different vmin. Repeate until self
consistent solution is found.

Currently this only functions for a single extended likelihood (CDMS-Si) and
 experiments capable of utilizing a poisson likelihood
"""


def Custom_SelfConsistent_Minimization(class_name, x0, mx, fp, fn, delta, vminStar=None,
                                       logetaStar=None,
                                       index=None, vmin_err=15.0, logeta_err=0.02):

        if vminStar is not None:
            vmin_list_reduced = x0[: len(x0)/2]
            vmin_list = np.sort(np.append(vmin_list_reduced, vminStar))

            index_hold = np.argwhere(vmin_list == vminStar)[0, 0]
            logeta_list_reduced = x0[len(x0)/2:]
            logeta_list = np.insert(logeta_list_reduced, index_hold, logetaStar)
        else:
            vmin_list = x0[: len(x0)/2]
            logeta_list = x0[len(x0)/2:]
            logeta_list_reduced = logeta_list
            vmin_list_reduced = vmin_list
            index_hold = None

        def constr_func_logeta(x):
            if logetaStar is not None:
                x = np.insert(x, index_hold, logetaStar)

            constraints = np.concatenate([-x, np.diff(-x)])

            is_not_close = np.logical_not(
                np.isclose(constraints, np.zeros_like(constraints), atol=1e-4))
            is_not_close[x.size] = True
            constr = np.where(is_not_close, constraints, np.abs(constraints))

            return constr

        def constr_func_vmin(x):
            if vminStar is not None:
                x = np.insert(x, index_hold, vminStar)

            constraints = np.concatenate([x, np.diff(x)])

            is_not_close = np.logical_not(
                np.isclose(constraints, np.zeros_like(constraints), atol=1e-4))
            is_not_close[x.size] = True
            constr = np.where(is_not_close, constraints, np.abs(constraints))

            return constr

        constr = ({'type': 'ineq', 'fun': constr_func_logeta})
        constr_vmin = ({'type': 'ineq', 'fun': constr_func_vmin})

        def optimize_logeta(logeta_list, vmin_list, class_name, mx, fp, fn, delta,
                            vminStar=None, logetaStar=None, index_hold=None):
            # CDMS-Si Likelihood
            optimize_func = class_name[0]._MinusLogLikelihood(np.append(vmin_list, logeta_list),
                                                              vminStar, logetaStar, index_hold)

            # Add in other likelihoods
            for i in range(1, len(class_name)):
                optimize_func += class_name[i]._MinusLogLikelihood(np.append(vmin_list, logeta_list),
                                                                   mx, fp, fn, delta, vminStar,
                                                                   logetaStar, index_hold)

            return optimize_func

        def minimize_over_vmin(vmin_list, logeta_list, class_name, mx, fp, fn, delta,
                               vminStar=None, logetaStar=None, index_hold=None):
            # CDMS-Si Likelihood
            optimize_func = class_name[0]._MinusLogLikelihood(np.append(vmin_list, logeta_list),
                                                              vminStar, logetaStar, index_hold)

            # Add in other likelihoods
            for i in range(1, len(class_name)):
                optimize_func += class_name[i]._MinusLogLikelihood(np.append(vmin_list, logeta_list),
                                                                   mx, fp, fn, delta, vminStar,
                                                                   logetaStar, index_hold)

            return optimize_func

        ni = 0
        check = False
        logeta_bnd = (-40.0, -12.0)
        bnd = [logeta_bnd] * vmin_list_reduced.size

        vmin_bnd = (0, 1000)
        bnd_vmin = [vmin_bnd] * vmin_list_reduced.size

        while ni < 10 and not check:

            mloglike_min = minimize(optimize_logeta, logeta_list_reduced,
                                    args=(vmin_list_reduced, class_name, mx,
                                          fp, fn, delta, vminStar,
                                          logetaStar, index_hold),
                                    method='SLSQP',
                                    bounds=bnd,
                                    constraints=constr)

            logeta_list_reduced = mloglike_min.x

            if logetaStar is not None:
                logeta_list_new = np.insert(logeta_list_reduced, index_hold, logetaStar)
            else:
                logeta_list_new = logeta_list_reduced

            vminloglike_min = minimize(minimize_over_vmin, vmin_list_reduced,
                                       args=(logeta_list_reduced, class_name,
                                       mx, fp, fn, delta, vminStar, logetaStar, index_hold),
                                       method='SLSQP', bounds=bnd_vmin, constraints=constr_vmin)

            vmin_list_reduced = vminloglike_min.x

            if vminStar is not None:
                vmin_list_new = np.sort(np.append(vmin_list_reduced, vminStar))
            else:
                vmin_list_new = vmin_list_reduced

            vmin_check = np.abs(np.asarray(vmin_list_new) - np.asarray(vmin_list))
            logeta_check = np.abs(np.asarray(logeta_list_new) - np.asarray(logeta_list))

            if np.amax(vmin_check) < vmin_err and np.amax(logeta_check) < logeta_err:
                print('n_iter', ni)
                print('Minimum Found')
                loglikeval = vminloglike_min.fun
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
                loglikeval = vminloglike_min.fun
            ni += 1

        if not check:
            print('Potential issue with convergence...')

        return np.concatenate([vmin_list_new, logeta_list_new]), loglikeval
