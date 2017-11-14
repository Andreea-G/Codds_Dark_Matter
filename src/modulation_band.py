from experiment_HaloIndep import *
from scipy import interpolate
from scipy.optimize import brentq, minimize, brute
from globalfnc import *
from basinhopping import *
from custom_minimizer import *
import matplotlib.pyplot as plt
import os   # for speaking
import parallel_map as par
from scipy.stats import poisson
import numpy.random as random
import numpy as np
from scipy.interpolate import interp1d, interpn, LinearNDInterpolator, RegularGridInterpolator
from scipy.integrate import quad, dblquad
import copy
import pymultinest

# TODO: This is only functioning for 2 bin analysis of DAMA...

class Experiment_EHI_Modulation(Experiment_HaloIndep):

    def __init__(self, expername, scattering_type, mPhi=mPhiRef, method='SLSQP', pois=False, gaus=False):
        super().__init__(expername, scattering_type, mPhi)
        module = import_file(INPUT_DIR + expername + ".py")

        self.BinData_C = module.BinData_C
        self.BinData_S = module.BinData_S
        self.BinEdges = module.BinEdges
        self.BinErr_C = module.BinError_C
        self.BinErr_S = module.BinError_S
        self.BinExp = module.Exposure
        self.Nbins = len(self.BinData_C) + len(self.BinData_S)
        self.Gaussian = gaus
        self.Poisson = pois
        self.Quenching = module.QuenchingFactor
        self.target_mass = module.target_nuclide_mass_list

        self.unique = True
        self.vmin_linspace_galactic = np.linspace(1., vesc, 250)
        self.vmin_max = self.vmin_linspace_galactic[-1]
        self.v_sun = np.array([11., 232., 7.])

    def v_Earth(self, times):
        ep1 = np.array([0.994, 0.1095, 0.003116])
        ep2 = np.array([-0.05173, 0.4945, -0.8677])
        vE = [29.8 * (np.cos(2.*np.pi*t)*ep1 + np.sin(2.*np.pi*t)*ep2) for t in times]
        return vE

    def _VMin_Guess(self):
        self.vmin_sorted_list = random.rand(int(self.Nbins), 3) * vesc

        return

    def _VMin_Guess_Constrained(self, vminStar):
        print('Looking for viable streams...')
        vmin_sorted_list = np.zeros((int(self.Nbins) + 1, 3))
        for i in range(self.Nbins / 2 + 1):
            found_it = False
            cnt = 0
            while not found_it:
                stream = random.rand(3) * vesc
                stream_E = stream - self.v_sun - self.v_Earth([0.])
                mag = np.sqrt(np.sum(stream_E * stream_E))
                if mag >= vminStar:
                    found_it = True
                    vmin_sorted_list[i] = stream
                else:
                    cnt += 1
                if cnt >= 1e4:
                    print('Could not find viable stream...')
                    exit()
        print('Found viable streams...')
        return vmin_sorted_list

    def CurH_Tab(self, mx, delta, fp, fn, file_output):
        if delta == 0:
            branches = [1]
        else:
            branches = [1, -1]


        self.response_tab = np.zeros((len(self.vmin_linspace), len(self.BinData_C)))
        self.curH_V = np.zeros((len(self.vmin_linspace)+1, len(self.BinData_C)))

        v_delta = min(VminDelta(self.mT, mx, delta))
        vmin_prev = 0
        curH_V_tab = np.zeros(len(self.BinData_C))

        for i, vmin in enumerate(self.vmin_linspace):
            print("vmin =", vmin)
            for j in range(len(self.BinEdges) - 1):
                for sign in branches:
                    self.response_tab[i, j] += self._Response_Finite(vmin, self.BinEdges[j],
                                                                     self.BinEdges[j+1], mx, fp, fn, delta)

                curH_V_tab[j] +=  np.trapz(self.response_tab[:i, j], self.vmin_linspace[:i]) / vmin
                vmin_prev = vmin
                self.curH_V[i+1, j] = curH_V_tab[j]

        self.curH_V[0, :] = 0.
        self.vmin_linspace = np.insert(self.vmin_linspace, 0, 0)
        file = file_output + "_CurH_Tab_V.dat"
        np.savetxt(file, self.curH_V)

        self.curh_interp = np.zeros(len(self.BinData_C), dtype=object)
        #self.curh_modamp_interp = np.zeros(len(self.BinData), dtype=object)
        for i in range(len(self.BinData_C)):
            self.curh_interp[i] = interp1d(self.vmin_linspace, self.curH_V[:, i], kind='cubic',
                                           bounds_error=False, fill_value=0)

        self.curH_V_modamp_C = np.zeros((len(self.BinData_C), self.vmin_linspace_galactic.shape[0]**3))
        self.curH_V_modamp_S = np.zeros((len(self.BinData_S), self.vmin_linspace_galactic.shape[0]**3))
        time_vals = np.linspace(0., 1., 100)
        for bin in range(len(self.BinData_C)):
            cnt = 0
            for i,ux in enumerate(self.vmin_linspace_galactic):
                for j,uy in enumerate(self.vmin_linspace_galactic):
                    for k,uz in enumerate(self.vmin_linspace_galactic):
                        speed = np.sqrt(np.sum((np.array([ux, uy, uz]) - self.v_sun -
                                                self.v_Earth(time_vals))**2., axis=1))

                        projection_c = 2.*np.trapz(self.curh_interp[bin](speed)*np.cos(2.*np.pi*time_vals), time_vals)
                        projection_s = 2.*np.trapz(self.curh_interp[bin](speed)*np.sin(2.*np.pi*time_vals), time_vals)
                        #print(speed, projection)
                        self.curH_V_modamp_C[bin, cnt] = projection_c
                        self.curH_V_modamp_S[bin, cnt] = projection_s
                        cnt += 1

            file_S = file_output + "_ModH_SIN_Tab_Bin_{:.0f}.dat".format(bin)
            file_C = file_output + "_ModH_COS_Tab_Bin_{:.0f}.dat".format(bin)
            np.savetxt(file_C, self.curH_V_modamp_C[bin])
            np.savetxt(file_S, self.curH_V_modamp_S[bin])

        return

    def CurH_Wrap(self, u1, u2, u3, bin, t):
        #uvel = np.array([u1, u2, u3]) - self.v_sun - self.v_Earth(t)
        mag_u = np.sqrt(u1**2. + u2**2. + u3**2.)
        return self.curh_interp[bin](mag_u)

    def ResponseTables(self, vmin_min, vmin_max, vmin_step, mx, fp, fn, delta,
                       output_file_tail):

        self.min_v_min = (self.target_mass + mx) / (mx * self.target_mass) * \
                         np.sqrt(2. * self.BinEdges[0] / self.QuenchingFactor(self.BinEdges[0]) *
                                 self.target_mass * 1e-6) * SpeedOfLight

        self.vmin_linspace = np.linspace(vmin_min, vmin_max, (vmin_max - vmin_min) / vmin_step + 1)

        self._VMin_Guess()
        file = output_file_tail + "_VminSortedList.dat"
        print(file)
        np.savetxt(file, self.vmin_sorted_list)

        self.CurH_Tab(mx, delta, fp, fn, output_file_tail)

        file = output_file_tail + "_VminLinspace.dat"
        print(file)
        np.savetxt(file, self.vmin_linspace)


        return

    def ImportResponseTables(self, output_file_tail, plot=False, pois=False):
        """ Imports the data for the response tables from files.
        """
        findmx = output_file_tail.find('_mx_')
        findgev = output_file_tail.find('GeV_')
        mx = float(output_file_tail[findmx + 4:findgev])
        self.min_v_min = (self.target_mass + mx) / (mx * self.target_mass) * \
                         np.sqrt(2. * self.BinEdges[0] / self.QuenchingFactor(self.BinEdges[0]) *
                                 self.target_mass * 1e-6) * SpeedOfLight


        self._VMin_Guess()
        file = output_file_tail + "_VminSortedList.dat"
        print(file)
        np.savetxt(file, self.vmin_sorted_list)

        file = output_file_tail + "_VminLinspace.dat"
        with open(file, 'r') as f_handle:
            self.vmin_linspace = np.loadtxt(f_handle)

        file = output_file_tail + "_CurH_Tab_V.dat"
        with open(file, 'r') as f_handle:
            self.curH_V = np.loadtxt(f_handle)

        self.curh_interp = np.zeros(len(self.BinData_C), dtype=object)
        self.curH_modamp_interp_C = np.zeros(len(self.BinData_C), dtype=object)
        self.curH_V_modamp_C = np.zeros((len(self.BinData_C), self.vmin_linspace_galactic.shape[0] ** 3))
        self.curH_modamp_interp_S = np.zeros(len(self.BinData_S), dtype=object)
        self.curH_V_modamp_S = np.zeros((len(self.BinData_S), self.vmin_linspace_galactic.shape[0] ** 3))
        for i in range(len(self.BinData_C)):
            self.curh_interp[i] = interp1d(self.vmin_linspace, self.curH_V[:, i], kind='cubic',
                                           bounds_error=False, fill_value=0)

            file_S = output_file_tail + "_ModH_SIN_Tab_Bin_{:.0f}.dat".format(i)
            file_C = output_file_tail + "_ModH_COS_Tab_Bin_{:.0f}.dat".format(i)
            loadf_S = np.loadtxt(file_S)
            loadf_C = np.loadtxt(file_C)
            self.curH_V_modamp_C[i] = loadf_C
            self.curH_V_modamp_S[i] = loadf_S

            num_vmin = len(self.vmin_linspace_galactic)
            self.curH_modamp_interp_C[i] = RegularGridInterpolator((self.vmin_linspace_galactic,
                                                                  self.vmin_linspace_galactic,
                                                                  self.vmin_linspace_galactic),
                                                              self.curH_V_modamp_C[i].reshape(num_vmin, num_vmin,
                                                                                              num_vmin),
                                                                 bounds_error=False,
                                                                 fill_value=0)
            self.curH_modamp_interp_S[i] = RegularGridInterpolator((self.vmin_linspace_galactic,
                                                                    self.vmin_linspace_galactic,
                                                                    self.vmin_linspace_galactic),
                                                                   self.curH_V_modamp_S[i].reshape(num_vmin, num_vmin,
                                                                                                   num_vmin),
                                                                   bounds_error=False,
                                                                   fill_value=0)

        return

    def del_curlH_modamp(self, bin, axis, stream, CorS='C', epsilon=2.):
        perturb = np.array([0., 0., 0.])
        perturb[axis] += epsilon
        v1 = stream + perturb
        v2 = stream - perturb
        if CorS == 'C':
            delcurlH = (self.curH_modamp_interp_C[bin](v1) - self.curH_modamp_interp_C[bin](v2)) / (2. * epsilon)
        else:
            delcurlH = (self.curH_modamp_interp_S[bin](v1) - self.curH_modamp_interp_S[bin](v2)) / (2. * epsilon)
        return delcurlH

    def rate_calculation(self, bin, streams, ln10_norms, CorS='C'):
        val = 0.
        for i,str in enumerate(streams):
            if CorS == 'C':
                val += np.power(10., ln10_norms[i]) * self.curH_modamp_interp_C[bin](str)
            else:
                val += np.power(10., ln10_norms[i]) * self.curH_modamp_interp_S[bin](str)
        return val

    def flat_prior(self, cube_val, cube_max=-1., cube_min=-50.):
        cube_val = cube_val * (cube_max - cube_min) + cube_min
        return cube_val

    def logflat_prior(self, cube_val, cube_max=533., cube_min=1.):
        power = (np.log10(cube_max) - np.log10(cube_min))*cube_val + np.log10(cube_min)
        cube_val = 10**power                                                                                        
        return cube_val
                                                                
    def prior_func(self, cube, ndim, nparams):
        params = self.param_names
        for i in range(ndim):
            if params[i] == "velocity":
                cube[i] = self.flat_prior(cube[i], cube_max=533., cube_min=0.)
                #cube[i] = self.logflat_prior(cube[i])
            elif params[i] == "mag":
                cube[i] = self.flat_prior(cube[i])
            else:
                cube[i] = self.flat_prior(cube[i], cube_max=10000., cube_min=-10000.)
            # print(params[i], cube[i])
        return

    def global_bestfit(self):
        """
        Returns maximum a posteriori values for parameters. 
        """
        samples = np.loadtxt(os.getcwd() + '/chains/1-post_equal_weights.dat')
        posterior = samples[:,-1]
        max_index = posterior.argmax()
        return samples[max_index,:-1]

    def MultiExperimentOptimalLikelihood(self, multiexper_input, class_name, mx, fp, fn, delta, output_file,
                                         output_file_CDMS, logeta_guess, nsteps_bin):



        #self.ImportResponseTables(output_file_CDMS, plot=False)
        streams = self.vmin_sorted_list
        #self.n_streams = len(self.vmin_sorted_list)
        self.n_streams = 1
        vars_guess = np.append(streams, logeta_guess * np.ones(self.n_streams))
        #print("vars_guess = ", vars_guess)

        self.param_names = []
        for i in range(self.n_streams):
            self.param_names.append("velocity")
            self.param_names.append("velocity")
            self.param_names.append("velocity")
        for i in range(self.n_streams):
            self.param_names.append("mag")

        pymultinest.run(self.loglike_total_multinest_wrapper, self.prior_func, len(self.param_names), resume=False,
                                                n_live_points=2000)
        bf_test = self.global_bestfit()
        print (bf_test)
        print (self.gaussian_m_ln_likelihood(bf_test))
        streams, norms = self.unpack_streams_norms(bf_test)
        #logeta_bnd = (-40.0, -15.0)
        #bnd_eta = [logeta_bnd] * self.n_streams
        #vmin_bnd = (0, self.vmin_max)
        #bnd_vmin = [vmin_bnd] * (self.n_streams * 3)
        #bnd = bnd_vmin + bnd_eta

        #opt = minimize(self.gaussian_m_ln_likelihood, vars_guess,
        #               jac=self.gaussian_jacobian,
        #               method='SLSQP',
        #               tol=1e-10, bounds=bnd,
        #               #constraints=constr,
        #               options={'maxiter': 300, 'disp': True})
        #print(opt)
        #print('\n')
        #
        #opt = minimize(self.gaussian_m_ln_likelihood, vars_guess,
        #               method='SLSQP', tol=1e-4,
        #               bounds=bnd, #constraints=constr,
        #               options={'maxiter': 200, 'disp': False})
        #print(opt)
        #streams, norms = self.unpack_streams_norms(opt.x)
        for i in range(int(self.Nbins / 2)):
            if (np.abs(self.rate_calculation(i, streams, norms, CorS='C') - self.BinData_C[i]) < 1e-3) and \
                    (np.abs(self.rate_calculation(i, streams, norms, CorS='S') - self.BinData_S[i]) < 1e-3):
                print('Bin {:.0f} is NOT Unique...'.format(i))
                self.unique = False
            print('Cos:', self.rate_calculation(i, streams, norms, CorS='C'), self.BinData_C[i])
            print('Sin:', self.rate_calculation(i, streams, norms, CorS='S'), self.BinData_S[i])

        file = output_file + "_GloballyOptimalLikelihood.dat"
        print(file)
        #np.savetxt(file, np.append([opt.fun], opt.x))
        np.savetxt(file, np.append([self.gaussian_m_ln_likelihood(bf_test)], bf_test))
        self.eta_BF_time_avg(bf_test, output_file)
        return

    def unpack_streams_norms(self, full_vec, add_streams=0):
        norms = full_vec[-(self.n_streams + add_streams):]
        streams = full_vec[:-(self.n_streams + add_streams)].reshape(self.n_streams + add_streams, 3)
        return streams, norms

    def loglike_total_multinest_wrapper(self, cube, ndim, nparams):
        streams_norms = np.zeros(ndim)
        for i in range(ndim):
            streams_norms[i] = cube[i]
        return -self.gaussian_m_ln_likelihood(streams_norms)
    
    def gaussian_m_ln_likelihood(self, streams_norms, vMinStar=None, etaStar=None, add_streams=0,
                                 include_penalty=False):
        # Note that I've had to insert penalty terms for the escape velocity and for constrained minimization
        
        streams, norms = self.unpack_streams_norms(streams_norms, add_streams=add_streams)
        m2_ln_like = 0.
        for i in range(int(self.Nbins / 2)):
            m2_ln_like += ((self.BinData_C[i] - self.rate_calculation(i, streams, norms, CorS='C')) / self.BinErr_C[i])**2.
            m2_ln_like += ((self.BinData_S[i] - self.rate_calculation(i, streams, norms, CorS='S')) / self.BinErr_S[i])**2.

        for str in streams:
            mag = np.sqrt(np.sum(str * str))
            if mag > vesc:
                m2_ln_like += (mag - vesc)**2. / 50.**2.

        if etaStar is not None:
            vh_bar = np.zeros(streams.shape[0])
            for j, str in enumerate(streams):
                time_arr = np.linspace(0., 1., 60)
                vh_bar[j] = self.v_bar_modulation(vMinStar, time_arr, str)
            eta_star = np.dot(vh_bar, np.power(10., norms))
            #print(np.log10(eta_star), etaStar)
            if include_penalty:
                if eta_star > 0.:
                    m2_ln_like += (np.log10(eta_star) - etaStar)**2. / (0.001)**2.
                else:
                    m2_ln_like += etaStar ** 2. / (0.001)**2.
            else:
                print('EtaStar: ', np.log10(eta_star), 'EtaStar Goal: ', etaStar)

        return m2_ln_like

    def gaussian_jacobian(self, streams_norms, vMinStar=None, etaStar=None, add_streams=0, include_penalty=False):
        streams, norms = self.unpack_streams_norms(streams_norms, add_streams=add_streams)
        stream_jac = np.zeros_like(streams)
        norms_jac = np.zeros_like(norms)

        for i in range(int(self.Nbins / 2)):
            for j in range(len(norms)):
                # print('test: ', self.del_curlH_modamp(i, 0, streams[j]), streams, norms)
                norms_jac[j] += -2. * (self.BinData_C[i] - self.rate_calculation(i, streams, norms, CorS='C')) / \
                                self.BinErr_C[i] ** 2. * self.curH_modamp_interp_C[i](streams[j]) *\
                                10.**norms[j] * np.log(10.)
                norms_jac[j] += -2. * (self.BinData_S[i] - self.rate_calculation(i, streams, norms, CorS='S')) / \
                                self.BinErr_S[i] ** 2. * self.curH_modamp_interp_S[i](streams[j]) * \
                                10. ** norms[j] * np.log(10.)
                stream_jac[j][0] += 2. * (self.BinData_C[i] - self.rate_calculation(i, streams, norms, CorS='C')) / \
                                self.BinErr_C[i] ** 2. * 10.**norms[j] * \
                                   self.del_curlH_modamp(i, 0, streams[j], CorS='C')
                stream_jac[j][0] += 2. * (self.BinData_S[i] - self.rate_calculation(i, streams, norms, CorS='S')) / \
                                    self.BinErr_S[i] ** 2. * 10. ** norms[j] * \
                                    self.del_curlH_modamp(i, 0, streams[j],  CorS='S')
                stream_jac[j][1] += 2. * (self.BinData_C[i] - self.rate_calculation(i, streams, norms, CorS='C')) / \
                                   self.BinErr_C[i] ** 2. * 10.**norms[j] * \
                                   self.del_curlH_modamp(i, 1, streams[j], CorS='C')
                stream_jac[j][1] += 2. * (self.BinData_S[i] - self.rate_calculation(i, streams, norms, CorS='S')) / \
                                    self.BinErr_S[i] ** 2. * 10. ** norms[j] * \
                                    self.del_curlH_modamp(i, 1, streams[j], CorS='S')
                stream_jac[j][2] += 2. * (self.BinData_C[i] - self.rate_calculation(i, streams, norms, CorS='C')) / \
                                   self.BinErr_C[i] ** 2. * 10.**norms[j] * \
                                   self.del_curlH_modamp(i, 2, streams[j], CorS='C')
                stream_jac[j][2] += 2. * (self.BinData_S[i] - self.rate_calculation(i, streams, norms, CorS='S')) / \
                                    self.BinErr_S[i] ** 2. * 10. ** norms[j] * \
                                    self.del_curlH_modamp(i, 2, streams[j], CorS='S')

        for j,str in enumerate(streams):
            mag = np.sqrt(np.sum(str * str))
            if mag > vesc:
                stream_jac[j][0] += 2. * (mag - vesc) / 50.**2. * str[0] / mag
                stream_jac[j][1] += 2. * (mag - vesc) / 50. ** 2. * str[1] / mag
                stream_jac[j][2] += 2. * (mag - vesc) / 50. ** 2. * str[2] / mag

        if etaStar is not None:
            vh_bar_N = np.zeros(streams.shape[0])
            vh_bar = np.zeros_like(streams)
            norms_star = np.zeros(streams.shape[0])
            coefC = np.sum(np.power(10., norms))
            farr = np.power(10., norms) / coefC
            for j, str in enumerate(streams):
                #stream_E = str - self.v_sun - self.v_Earth([0.])
                time_arr = np.linspace(0., 1., 60)
                vh_bar[j][0] = np.power(10., norms[j]) * self.v_bar_modulation_jac(vMinStar, time_arr, str, 0)
                vh_bar[j][1] = np.power(10., norms[j]) * self.v_bar_modulation_jac(vMinStar, time_arr, str, 1)
                vh_bar[j][2] = np.power(10., norms[j]) * self.v_bar_modulation_jac(vMinStar, time_arr, str, 2)
                norms_star[j] = -np.power(10., norms[j]) * np.log(10.) * self.v_bar_modulation_jac(vMinStar, time_arr,
                                                                                               str, 0, epsilon=0)
                vh_bar_N = np.power(10., norms[j]) * self.v_bar_modulation(vMinStar, time_arr, str)

            for j in range(len(norms)):
                norms_jac[j] += 2. * (np.log10(vh_bar_N) - etaStar) / (0.001**2.) * norms_star[j]
                stream_jac[j][0] += 2. * (np.log10(vh_bar_N) - etaStar) / (0.001 ** 2.) * vh_bar[j][0]
                stream_jac[j][1] += 2. * (np.log10(vh_bar_N) - etaStar) / (0.001 ** 2.) * vh_bar[j][1]
                stream_jac[j][2] += 2. * (np.log10(vh_bar_N) - etaStar) / (0.001 ** 2.) * vh_bar[j][2]

        # print('Jac: ', stream_jac, norms_jac)
        return np.append(stream_jac.flatten(), norms_jac).flatten()

    def constrained_gaussian_m_ln_likelihood(self, variables, vMinStar=None, etaStar=None, add_streams=0,
                                             p_etastar=False):
        # Note that I've had to insert penalty terms for the escape velocity and for constrained minimization
        streams_norms = variables[:-1]
        lag_mult = variables[-1]

        streams, norms = self.unpack_streams_norms(streams_norms, add_streams=add_streams)
        m2_ln_like = 0.
        for i in range(int(self.Nbins / 2)):
            m2_ln_like += ((self.BinData_C[i] - self.rate_calculation(i, streams, norms, CorS='C')) / self.BinErr_C[i])**2.
            m2_ln_like += ((self.BinData_S[i] - self.rate_calculation(i, streams, norms, CorS='S')) / self.BinErr_S[i]) ** 2.

        for str in streams:
            mag = np.sqrt(np.sum(str * str))
            if mag > vesc:
                m2_ln_like += (mag - vesc)**2. / 50.**2.


        vh_bar = np.zeros(streams.shape[0])
        for j, str in enumerate(streams):
            time_arr = np.linspace(0., 1., 60)
            vh_bar[j] = self.v_bar_modulation(vMinStar, time_arr, str)

        eta_star = np.dot(vh_bar, np.power(10., norms))
        #print(np.log10(eta_star), etaStar)
        if eta_star <= 0.:
            eta_star = 1e-50
        m2_ln_like -= lag_mult * np.abs(np.log10(eta_star) - etaStar)
        #print('Lag', lag_mult, m2_ln_like)
        if p_etastar:
            print('EtaStar: ', np.log10(eta_star), 'EtaStar Goal: ', etaStar)

        return m2_ln_like

    def constrained_gaussian_jacobian(self, variables, vMinStar=None, etaStar=None, add_streams=0,
                                      p_etastar=False):
        streams_norms = variables[:-1]
        lag_mult = variables[-1]

        streams, norms = self.unpack_streams_norms(streams_norms, add_streams=add_streams)
        stream_jac = np.zeros_like(streams)
        norms_jac = np.zeros_like(norms)
        m2_ln_like = 0.
        
        stream_pert = 5.
        norm_pert = 0.001
        
        for i in range(int(self.Nbins / 2)):
            for j,stre in enumerate(streams):
                rc_cos = self.rate_calculation(i, [stre], [norms[j]], CorS='C')
                rc_sin = self.rate_calculation(i, [stre], [norms[j]], CorS='S')
                
                
                strm_ep = stre + np.array([stream_pert, 0., 0.])
                rfake_cos = self.rate_calculation(i, [strm_ep], [norms[j]], CorS='C')
                rfake_sin = self.rate_calculation(i, [strm_ep], [norms[j]], CorS='S')
                m2_ln_like += ((rfake_cos**2. - rc_cos**2. + 2.*self.BinData_C[i] * 
                               (rc_cos - rfake_cos)) / (stream_pert * self.BinErr_C[i])**2.)**2.
                m2_ln_like += ((rfake_sin**2. - rc_sin**2. + 2.*self.BinData_S[i] * 
                               (rc_sin - rfake_sin)) / (stream_pert * self.BinErr_S[i])**2.)**2.
                               
                strm_ep = stre + np.array([0., stream_pert, 0.])
                rfake_cos = self.rate_calculation(i, [strm_ep], [norms[j]], CorS='C')
                rfake_sin = self.rate_calculation(i, [strm_ep], [norms[j]], CorS='S')
                m2_ln_like += ((rfake_cos**2. - rc_cos**2. + 2.*self.BinData_C[i] * 
                               (rc_cos - rfake_cos)) / (stream_pert * self.BinErr_C[i])**2.)**2.
                m2_ln_like += ((rfake_sin**2. - rc_sin**2. + 2.*self.BinData_S[i] * 
                               (rc_sin - rfake_sin)) / (stream_pert * self.BinErr_S[i])**2.)**2.
                               
                strm_ep = stre + np.array([0., 0., stream_pert])
                rfake_cos = self.rate_calculation(i, [strm_ep], [norms[j]], CorS='C')
                rfake_sin = self.rate_calculation(i, [strm_ep], [norms[j]], CorS='S')
                m2_ln_like += ((rfake_cos**2. - rc_cos**2. + 2.*self.BinData_C[i] * 
                               (rc_cos - rfake_cos)) / (stream_pert * self.BinErr_C[i])**2.)**2.
                m2_ln_like += ((rfake_sin**2. - rc_sin**2. + 2.*self.BinData_S[i] * 
                               (rc_sin - rfake_sin)) / (stream_pert * self.BinErr_S[i])**2.)**2.
                
                #print('Str Contrib: ', m2_ln_like)
                rfake_cos = self.rate_calculation(i, [stre], [norms[j]+norm_pert], CorS='C')
                rfake_sin = self.rate_calculation(i, [stre], [norms[j]+norm_pert], CorS='S')
                m2_ln_like += ((rfake_cos**2. - rc_cos**2. + 2.*self.BinData_C[i] * 
                               (rc_cos - rfake_cos)) / (norm_pert * self.BinErr_C[i])**2.)**2.
                m2_ln_like += ((rfake_sin**2. - rc_sin**2. + 2.*self.BinData_S[i] * 
                               (rc_sin - rfake_sin)) / (norm_pert * self.BinErr_S[i])**2.)**2.
                
        rate_contrib = copy.copy(m2_ln_like)
        print('Rate Contrib: ', -np.sqrt(rate_contrib))
                
        vh_bar_N = np.zeros(streams.shape[0])
        vh_bar = np.zeros_like(streams)
        vh_bar_per = np.zeros_like(streams)
        for j,str in enumerate(streams):
            mag = np.sqrt(np.sum(str * str))
            str2 = str + np.array([stream_pert, 0., 0.])
            mag2 = np.sqrt(np.sum(str2 * str2))
            val1 = 0.
            val2 = 0.
            if mag > vesc:
                val1 = (mag - vesc)**2. / 50.**2.
            if mag2 > vesc:
                val2 = (mag2 - vesc)**2. / 50.**2.
            m2_ln_like += ((val2**2. - val1**2. + vesc*(val1 - val2)) / stream_pert **2.)**2.

            time_arr = np.linspace(0., 1., 60)
            strm_ep = stre + np.array([stream_pert, 0., 0.])
            vh_bar[j][0] = self.v_bar_modulation(vMinStar, time_arr, str)
            vh_bar_per[j][0] = self.v_bar_modulation(vMinStar, time_arr, str+strm_ep)
            
            strm_ep = stre + np.array([0., stream_pert, 0.])
            vh_bar[j][1] = self.v_bar_modulation(vMinStar, time_arr, str)
            vh_bar_per[j][1] = self.v_bar_modulation(vMinStar, time_arr, str+strm_ep)
            
            strm_ep = stre + np.array([0., 0., stream_pert])
            vh_bar[j][2] = self.v_bar_modulation(vMinStar, time_arr, str)
            vh_bar_per[j][2] = self.v_bar_modulation(vMinStar, time_arr, str+strm_ep)
          
            
            vh_bar_N[j] = self.v_bar_modulation(vMinStar, time_arr, str)
            if vh_bar_N[j] == 0:
                vh_bar_N[j] = 1e-30
                
        eta_calc = np.dot(vh_bar_N, np.power(10., norms))
        m2_ln_like += (np.log10(eta_calc) - etaStar)**2.
        for j,str in enumerate(streams):
            m2_ln_like += (lag_mult * np.power(10., norms[j])* (vh_bar_per[j][0] - vh_bar[j][0]) / (np.log(10.) * eta_calc * stream_pert))**2.
            m2_ln_like += (lag_mult * np.power(10., norms[j])* (vh_bar_per[j][1] - vh_bar[j][1]) / (np.log(10.) * eta_calc * stream_pert))**2.
            m2_ln_like += (lag_mult * np.power(10., norms[j])* (vh_bar_per[j][2] - vh_bar[j][2]) / (np.log(10.) * eta_calc * stream_pert))**2.
            m2_ln_like += (lag_mult)**2.
        
        print('Lambda Term Contrib: ', -np.sqrt(m2_ln_like - rate_contrib))
        print('Eta Star Calc:', np.log10(eta_calc), ' Eta Star: ', etaStar, ' Val: ', -np.sqrt(m2_ln_like))
      
        return np.sqrt(m2_ln_like)


    def ImportMultiOptimalLikelihood(self, output_file, output_file_CDMS, plot=False):
        #self.ImportResponseTables(output_file_CDMS, plot=False)
        file = output_file + "_GloballyOptimalLikelihood.dat"
        with open(file, 'r') as f_handle:
            optimal_result = np.loadtxt(f_handle)
        self.max_like = optimal_result[0]
        self.n_streams = 1
        
        self.streams, self.norms = self.unpack_streams_norms(optimal_result[1:])
        return


    def PlotQ_KKT_Multi(self, class_name, mx, fp, fn, delta, output_file, kkpt):
        return
    
    def loglike_total_multinest_wrapper_constr(self, cube, ndim, nparams):
        streams_norms = np.zeros(ndim)
        for i in range(ndim):
            streams_norms[i] = cube[i]
        #return -self.constrained_gaussian_m_ln_likelihood(streams_norms, self.constr_info['vstar'], 
        #                                                  self.constr_info['letastar'], 1, True)
        #return -self.gaussian_m_ln_likelihood(streams_norms, vMinStar=self.constr_info['vstar'],
        #                                      etaStar=self.constr_info['letastar'], add_streams=1,
        #                                      include_penalty=True)
        return -self.constrained_gaussian_jacobian(streams_norms, vMinStar=self.constr_info['vstar'], 
                                                  etaStar=self.constr_info['letastar'], add_streams=1,
                                                  p_etastar=True)
    
    def MultiExperConstrainedOptimalLikelihood(self, vminStar, logetaStar, multiexper_input, class_name,
                                               mx, fp, fn, delta, plot=False, leta_guess=-26.):
        #streams = self._VMin_Guess_Constrained(vminStar)
        
        #self.ImportResponseTables(output_file_CDMS, plot=False)
        n_streams = self.n_streams + 1
        #vars_guess = np.append(streams, leta_guess * np.ones(n_streams))
        #vars_guess = np.append(vars_guess, np.array([1.]))
        #print("vars_guess = ", vars_guess)
        
        #logeta_bnd = (-60.0, -10.0)
        #bnd_eta = [logeta_bnd] * n_streams
        #vmin_bnd = (0, self.vmin_max)
        #bnd_vmin = [vmin_bnd] * (n_streams * 3)
        #bnd = bnd_vmin + bnd_eta + [(-100., 100.)]
        #constr = ({'type':'ineq', 'fun': lambda x: self.constr_func_Constrained(x, vminStar,
        #                                                                       logetaStar, add_streams=1)})


        #opt = minimize(self.constrained_gaussian_m_ln_likelihood, vars_guess,
        #               method='SLSQP', tol=1e-8,
        #               jac=self.constrained_gaussian_jacobian,
        #               bounds=bnd,  args=(vminStar, logetaStar, 1, False), #constraints=constr,
        #               options={'maxiter': 1000, 'disp': False})
        #print(opt)

        self.param_names = []
        for i in range(n_streams):
            self.param_names.append("velocity")
            self.param_names.append("velocity")
            self.param_names.append("velocity")
        for i in range(n_streams):
            self.param_names.append("mag")
            
        self.param_names.append("X")
        self.constr_info = {'vstar': vminStar, 'letastar': logetaStar}
        
        pymultinest.run(self.loglike_total_multinest_wrapper_constr, self.prior_func, 
                        len(self.param_names), resume=False, n_live_points=2000)
        bf_test = self.global_bestfit()
        print (bf_test)
        
        val = self.constrained_gaussian_m_ln_likelihood(bf_test, vMinStar=vminStar, etaStar=logetaStar,
                                            add_streams=1, p_etastar=True)

        streams, norms = self.unpack_streams_norms(bf_test[:-1], add_streams=1)
        print(val, streams, norms)
        return val

    def eta_BF_time_avg(self, streams_norms, output_file):
        streams, norms = self.unpack_streams_norms(streams_norms)
        coefC = np.sum(np.power(10., norms))
        farr = np.power(10., norms) / coefC

        time_arr = np.linspace(0., 1., 60)
        vmin_arr = np.logspace(0., np.log10(800), 100)
        vh_bar = np.zeros((len(vmin_arr), streams.shape[0]))

        for i,vmin in enumerate(vmin_arr):
            for j,str in enumerate(streams):
                vh_bar[i, j] = self.v_bar_modulation(vmin, time_arr, str)
        eta_0_bf = coefC * np.dot(vh_bar, farr)

        vmin_arr = np.insert(vmin_arr, 0, 0)
        eta_0_bf = np.insert(eta_0_bf, 0, eta_0_bf[0])
        file_nme = output_file + '_TimeAverage_Eta_BF.dat'
        np.savetxt(file_nme, np.column_stack((vmin_arr, eta_0_bf)))
        return

    def v_bar_modulation(self, vmin, tarray, stream):
        speed = np.sqrt(np.sum((stream - self.v_sun - self.v_Earth(tarray)) ** 2., axis=1))
        #print('Speed: ', speed, 'VStar: ', vmin)
        integrnd = 1. / speed
        integrnd[speed < vmin] = 0.
        val_v = np.trapz(integrnd, tarray)
        if val_v > 0.:
            return val_v
        else:
            return 0.

    def v_bar_modulation_jac(self, vmin, tarray, stream, comp, epsilon=5):
        k = 0.2
        perturb = np.zeros(3)
        perturb[comp] += epsilon
        str = stream + perturb - self.v_sun - self.v_Earth(tarray)
        speed = np.sqrt(np.sum(str * str, axis=1))
        integrnd = 1. / speed * 0.5 * (1. + np.tanh(k * (speed - vmin)))
        val_v = np.trapz(integrnd, tarray)
        return val_v

    def constr_func_Constrained(self, streams_norms, vstar, etaStar, add_streams=1):
        streams, norms = self.unpack_streams_norms(streams_norms, add_streams=add_streams)
        time_arr = np.linspace(0., 1., 60)
        coefC = np.sum(np.power(10., norms))
        farr = np.power(10., norms) / coefC
        vh_bar = np.zeros(streams.shape[0])
        for j,str in enumerate(streams):
            vh_bar[j] = self.v_bar_modulation(vstar, time_arr, str)
        #print(vh_bar)
        #print(farr)
        #print(np.dot(vh_bar, farr))
        eta_star = coefC * np.dot(vh_bar, farr)
        #print(np.log10(eta_star), etaStar)
        if eta_star <= 0.:
            return -1.
        if np.abs(np.log10(eta_star) - etaStar) < 2e-2:
            return 1.
        else:
            return -1.

    def VminSamplingList(self, output_file_tail, output_file_CDMS, vmin_min, vmin_max, vmin_num_steps,
                         steepness_vmin=1.5, steepness_vmin_center=2.5, MULTI_EXPER=False,
                         plot=False):
        self.ImportMultiOptimalLikelihood(output_file_tail, output_file_CDMS)

        self.vmin_sampling_list = np.logspace(np.log10(vmin_min), np.log10(vmin_max), vmin_num_steps)

        xmin = vmin_min
        xmax = vmin_max

        x_num_steps = vmin_num_steps  # + 4
        s = steepness_vmin
        sc = steepness_vmin_center

        x_lin = np.linspace(xmin, xmax, 1000)
        self.optimal_vmin = np.zeros(self.Nbins)
        for i in range(self.Nbins):
            vmin_vals = self.streams[i] - self.v_sun - self.v_Earth([0.21])
            self.optimal_vmin[i] = np.sqrt(np.sum(vmin_vals*vmin_vals))
        self.args_best_fit_sort = np.argsort(self.optimal_vmin)
        self.optimal_vmin = self.optimal_vmin[self.args_best_fit_sort]
        numx0 = self.optimal_vmin.size

        print("x0 =", self.optimal_vmin)

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

        def g_total(x, sign=1, x0=self.optimal_vmin, s_list=s_list):
            return np.array([sign * g(x, x0[i], s_list[i, 0], s_list[i, 1])
                             for i in range(x0.size)]).prod(axis=0)

        g_lin = g_total(x_lin)

        xT_guess = (self.optimal_vmin[:-1] + self.optimal_vmin[1:]) / 2
        bounds = np.array([(self.optimal_vmin[i], self.optimal_vmin[i + 1])
                           for i in range(self.optimal_vmin.size - 1)])
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
        self.optimal_logeta = self.norms[self.args_best_fit_sort]
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
                if logeta_opt < -40:
                    logeta_opt = -31.
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
            return -constr_list.sum() * 10 ** 2
        return logL_interp(vars_list)

    def ConfidenceBand(self, output_file_tail, delta_logL, interpolation_order,
                       extra_tail="", multiplot=False):
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
            x = table[:, 0]  # this is logeta
            y = table[:, 1]  # this is logL
            if self.Poisson:
                x = x[y < 1e5]
                y = y[y < 1e5]
            logL_interp = interpolate.interp1d(x, y, kind='cubic', bounds_error=False, fill_value=1e5)

            def _logL_interp(vars_list, constraints):
                constr_not_valid = constraints(vars_list)[:-1] < 0
                if np.any(constr_not_valid):
                    constr_list = constraints(vars_list)[constr_not_valid]
                    return -constr_list.sum() * 1e2
                return logL_interp(vars_list)

            print(self.optimal_logL - delta_logL)
            print(np.array([table[0, 0]]), " ", table[-1, 0])
            print(logeta_optim)

            if self.Poisson and logeta_optim < -35:
                logeta_optim = -30.

            def constr_func(logeta, logeta_min=np.array([table[0, 0]]),
                            logeta_max=np.array([table[-1, 0]])):
                return np.concatenate([logeta - logeta_min, logeta_max - logeta])

            constr = ({'type': 'ineq', 'fun': constr_func})
            try:
                # logeta_minimLogL = minimize(_logL_interp, np.array([logeta_optim]),
                #                             args=(constr_func,), constraints=constr).x[0]
                logeta_minimLogL = minimize(logL_interp, np.array([logeta_optim])).x[0]
                # logeta_minimLogL = x[np.argmin(y)]
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
                if y[
                    0] > self.optimal_logL + delta_logL:  # and abs(logeta_minimLogL) < self.optimal_logL + delta_logL:
                    sol = brentq(lambda logeta: logL_interp(logeta) - self.optimal_logL -
                                                delta_logL,
                                 x[0], logeta_minimLogL)

                    self.vmin_logeta_band_low += \
                        [[self.vmin_sampling_list[index], sol]]
                else:
                    self.vmin_logeta_band_low += \
                        [[self.vmin_sampling_list[index], -40.]]
            except ValueError:
                print("ValueError: Error in calculating vmin_logeta_band_low")
                error = T

            try:
                if (y[-1] > self.optimal_logL + delta_logL) and \
                        (logL_interp(logeta_minimLogL) < self.optimal_logL + delta_logL):
                    print(logeta_minimLogL, x[-1], logL_interp(logeta_minimLogL), self.optimal_logL + delta_logL)

                    sol = brentq(lambda logeta: logL_interp(logeta) - self.optimal_logL -
                                                delta_logL, logeta_minimLogL, x[-1])
                    self.vmin_logeta_band_up += \
                        [[self.vmin_sampling_list[index], sol]]

            except ValueError:
                print("ValueError: Error in calculating vmin_logeta_band_hi")

                error = T

            if False:
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




