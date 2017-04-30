"""
Copyright (c) 2015 Andreea Georgescu

Created on Sat Dec  6 15:32:34 2014

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

from experiment_HaloIndep_Band import *
from collections import defaultdict
from scipy.interpolate import interp1d
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.ticker import FixedLocator, ScalarFormatter, FuncFormatter
from matplotlib import rc, rcParams
import os
rcParams['text.usetex'] = True  # needed to use LaTeX for labels and legends
rcParams['text.latex.preamble'] = [r'\boldmath']  # needed for bold labels and legends


class PlotData:
    """ Class to plot the data.
    Input:
        exper_name: string
            Name of experiment.
        HALO_DEP: bool
            Whether the analysis is halo-dependent or halo-independent.
        plot_close: bool, optional
            Whether the plot should be cleared and started from scratch, or new limits
            should be added to a previous plot.
    """
    count = defaultdict(lambda: -1)
    exper_plotted = set()

    def __init__(self, exper_name, HALO_DEP, plot_close=True):
        self.exper_name = exper_name
        self.HALO_DEP = HALO_DEP
        PlotData.exper_plotted.add(self.exper_name)

        if plot_close:
            plt.close()

        # set aspect ratio
        plt.rcParams["figure.figsize"][1] = plt.rcParams["figure.figsize"][0]

        self.set_axes_labels()
        self.set_ticks()

    @staticmethod
    def reset():
        PlotData.count = defaultdict(lambda: -1)
        PlotData.exper_plotted = set()
        plt.close()

    @staticmethod
    def make_legend(HALO_DEP, scattering_type, mPhi, fp, fn, delta, mx=None,
                    log_sigma_p=None, fontsize=13):
        rc('font', family='serif', weight='extra bold')
        # parameters
        legends = []
        if HALO_DEP:
            legends.extend(["SHM",
                            "$v_\odot$ = " + str(round(vobs)) + " km/s",
                            "$v_0$ = " + str(round(v0bar)) + " km/s",
                            "$v_\mathrm{esc}$ = " + str(round(vesc)) + " km/s",
                            ""])
        if mPhi == mPhiRef:
            legends.append(r"Contact")
        else:
            legends.append(r"${m_\phi = " + str(round(mPhi)) + r"}$ MeV")
        legends.append(r"${\delta = " + str(int(delta)) + r"}$ keV")
        if mx is not None:
            legends.append(r"${m = " + str(mx) + r"}$ GeV")
        if scattering_type == "SI":
            legends.append(r"${f_n/f_p = " + str(fn/fp) + r"}$")
        else:
            legends.append(r"${a_n/a_p = " + str(fn/fp) + r"}$")

        legend_param = plt.legend(legends, loc=1, frameon=False,
                                  prop={'size': fontsize},
                                  handlelength=0, handletextpad=0)
        for item in legend_param.legendHandles:
            item.set_visible(False)

        # experiments
        legends = [name for name in legend_names
                   if np.any([exp in PlotData.exper_plotted
                              for exp in legend_names[name]])]
        colors = [Color[legend_names[name][0]] for name in legends]
        if log_sigma_p is not None:
            legends = [name if "SHM" not in name
                       else name.replace("-40", str(log_sigma_p))
                       for name in legends]
        legends = [r"\textbf{" + name + "}" for name in legends]
        legend_exper = plt.legend(legends, loc=3, frameon=False,
                                  prop={'size': fontsize},
                                  handlelength=0, handletextpad=0)
        for text, color in zip(legend_exper.get_texts(), colors):
            plt.setp(text, color=color)
        for item in legend_exper.legendHandles:
            item.set_visible(False)
        plt.gca().add_artist(legend_param)

    def get_linestyle(self):
        dashes = line_dashes.get(self.exper_name, None)
        PlotData.count[self.exper_name] += 1
        linestyle = linestyles[PlotData.count[self.exper_name] % len(linestyles)]
        return linestyle, dashes

    def set_axes_labels(self, fontsize=20):
        """ Set axis labels, depending on whether it is for halo-dependent or not.
        """
        rc('font', family='serif', size=fontsize, weight='bold')
        if self.HALO_DEP:
            plt.xlabel(r'$m$ [GeV]')
            plt.ylabel(r'$\sigma_p$ [cm$^2$]')
            plt.axes().set_xscale('log')
            plt.axes().xaxis.set_major_formatter(ScalarFormatter())
        else:
            plt.minorticks_on()
            plt.xlabel(r'$v_\mathrm{min}$ [km/s]')
            plt.ylabel(r'$\eta$ $\rho$ $\sigma_p / m$ $[$days$^{-1}]$')

    def set_ticks(self):
        # set major ticks on yaxis
        def major_formatter(x, pos):
            return "$10^{{{0}}}$".format(int(x))

        plt.axes().yaxis.set_major_formatter(FuncFormatter(major_formatter))

        # set minor ticks on yaxis
        minor_ticks = [np.log10(i) for i in range(2, 10)]
        minor_ticks = [i + j for i in minor_ticks for j in range(-100, 0)]
        plt.axes().yaxis.set_minor_locator(FixedLocator(minor_ticks))

        # set ticks size
        plt.axes().xaxis.set_tick_params(length=8, width=1.5, which='major')
        plt.axes().xaxis.set_tick_params(length=4, width=1.5, which='minor')
        plt.axes().yaxis.set_tick_params(length=8, width=1.5, which='major')
        plt.axes().yaxis.set_tick_params(length=4, width=1.5, which='minor')

    def plot_limits(self, upper_limit, kind, linewidth, linestyle, dashes, plot_dots,
                    alpha=1):
        """ Make a list of the x and y coordinates of the plots, and plot them.
        """
        
        if upper_limit.size == 0:   # nothing to plot
            print("upper_limit is empty!")
            return None, None
        elif upper_limit.ndim == 1:  # only one point, so no interpolation
            if self.HALO_DEP:
                x = [10**upper_limit[0]]
            else:
                x = [upper_limit[0]]
            y = [upper_limit[1]]
            plt.plot(x, y, "o", alpha=alpha)
            return x, y
        else:   # more than one point, so decide on the interpolation order and plot
            if self.HALO_DEP:
                x = 10**upper_limit[:, 0]
            else:
                x = upper_limit[:, 0]
            y = upper_limit[:, 1]
            num_points = x.size
            if num_points == 2 or kind == "linear":
                interp_kind = "linear"
            elif num_points == 3 or kind == "quadratic":
                interp_kind = "quadratic"
            else:
                interp_kind = "cubic"
            # print('Interpolation',interp_kind)
            interp = interp1d(x, y, kind=interp_kind)
            x1 = np.linspace(x[0], x[-1], 1000)
            
            if plot_dots:
                plt.plot(x, y, "o")
            if dashes is not None:
                plt.plot(x1, interp(x1), dashes=dashes, linewidth=linewidth,
                         color=Color[self.exper_name.split()[0]], alpha=alpha)
            else:
                plt.plot(x1, interp(x1), linestyle=linestyle,  linewidth=linewidth,
                         color=Color[self.exper_name.split()[0]], alpha=alpha)
            return x1, interp(x1)

    def plot_crosses(self, crosses, linewidth=3, plot_show=True, alpha=1):
#        int_resp = crosses[0]

        def xy_points(indices):
            x = crosses[indices[0]]
            xerr_left, xerr_right = crosses[indices[1]], crosses[indices[2]]
            y = np.log10(crosses[indices[3]])
            yerr_up = np.log10(crosses[indices[3]] + crosses[indices[4]]) - y

            if "DAMA" in self.exper_name:
                yerr_low = y - np.log10(crosses[indices[3]] - crosses[indices[5]])
                yerr_low = np.array([ye if ye == ye else 100
                                     for ye in yerr_low])
            elif self.exper_name == "CDMSSi2012":
                yerr_low = - (y - np.log10(crosses[indices[3]] + crosses[indices[5]]))

            xerr = [xerr_left, xerr_right]
            yerr = [yerr_low, yerr_up]
            return x, y, xerr, yerr

        if self.exper_name == "CDMSSi2012":
            indices = [1, 2, 3, 4, 5, 5]
        elif "DAMA" in self.exper_name:
            indices = [1, 2, 3, 4, 5, 5]
        else:
            print('No crosses available for this experiment')
            return

        x, y, xerr, yerr = xy_points(indices)
        print(x, y, xerr, yerr)
        plt.errorbar(x, y, xerr=xerr, yerr=yerr, ecolor=Color[self.exper_name],
                     alpha=alpha,  linewidth=0, elinewidth=3,
                     capsize=linewidth + 1, capthick=linewidth)

        if plot_show:
            plt.show()

    def __call__(self, upper_limit, lower_limit=None, kind=None, linewidth=3,
                 fill=True, alpha=1, plot_dots=False, plot_show=False):
        """ Make plots for the upper limits.
        Input:
            upper_limit: list of lists
                List of x and y coordinates for points representing the upper limit.
            lower_limit: list of lists, optional
                List of x and y coordinates for points representing the lower limit.
            kind: string, optional
                The interpolation kind: 'linear', 'quadratic' or 'cubic'.
            linewidth: float, optional
                Width of the plotted line.
            fill: bool, optional
                If there is a lower_limit, fill between the upper and lower limit.
            alpha: float, optional
                Transparency parameter from 0 to 1.
            plot_dots: bool, optional
                Whether the plot should show the data points or just the interpolation.
            plot_show: bool, optional
                Whether the plot should be shown or not.
        """
        linestyle, dashes = self.get_linestyle()
        
        
        if lower_limit is not None and fill:
            linewidth = 0
        #print('Upper/LowerLimit: ', upper_limit, lower_limit)
        x_upper, y_upper = self.plot_limits(upper_limit, kind, linewidth, linestyle,
                                            dashes, plot_dots, alpha=alpha)
        if lower_limit is not None:
            x_lower, y_lower = self.plot_limits(lower_limit, kind, linewidth, linestyle,
                                                dashes, plot_dots, alpha=alpha)
            if fill:
                plt.fill_between(x_lower, y_lower, y_upper, where=y_upper>y_lower,
                                 color=Color[self.exper_name.split()[0]], alpha=alpha)

        if plot_show:
            plt.show()


class RunProgram:
    """ Class implementing the main run of the program.
    """
    def init_experiment(self, exper_name, scattering_type, mPhi, delta, HALO_DEP,
                        EHI_METHOD, MAKE_CROSSES, MAKE_LIMITS, log_sigma_p, quenching):
        """Select which experiment class we must use, depending on what statistical
        analysis we need, and initialize the experiment.
        """
        self.exper_name = exper_name
        print('name = ', exper_name)
        if HALO_DEP:
            print('Halo Dependent')
            if exper_name in MaximumGapLimit_exper:
                class_name = MaxGapExperiment
            elif exper_name in GaussianLimit_exper:
                class_name = GaussianExperiment
            elif exper_name in Poisson_exper:
                class_name = PoissonExperiment
            elif exper_name in BinnedSignal_exper:
                class_name = DAMAExperiment
            elif exper_name.split()[0] in BinnedSignal_exper:
                class_name = DAMAExperimentCombined
            elif exper_name in DAMALimit_exper:
                class_name = DAMATotalRateExperiment
            elif exper_name in Poisson_likelihood:
                class_name = PoissonLikelihood
            elif exper_name in Extended_Like:
                class_name = Extended_likelihood
            else:
                print("NotImplementedError: This experiment was not implemented!")
                return
        else:
            print('Halo Independent')
            if exper_name in EHImethod_exper and np.any(EHI_METHOD):
                print('EHI Method')
                class_name = Experiment_EHI
            elif exper_name in MaximumGapLimit_exper and MAKE_LIMITS:
                class_name = MaxGapExperiment_HaloIndep
            elif exper_name in Poisson_exper and MAKE_LIMITS:
                class_name = PoissonExperiment_HaloIndep
            elif exper_name in GaussianLimit_exper and MAKE_LIMITS:
                class_name = GaussianExperiment_HaloIndep
            elif exper_name in Crosses_exper and MAKE_CROSSES:
                class_name = Crosses_HaloIndep
            elif exper_name.split()[0] in Crosses_exper and MAKE_CROSSES:
                class_name = Crosses_HaloIndep_Combined
            elif exper_name in Poisson_Like and MAKE_LIMITS:
                class_name = Poisson_Likelihood
            elif exper_name in SHM_line:
                class_name = Standard_Halo_Model
            else:
                print("NotImplementedError: This experiment was not implemented!")
                return
            # if delta > 0 we have to use the integration in recoil energy ER
            if delta > 0:
                class_name.__bases__ = (Experiment_HaloIndep_ER,)

        if exper_name in SHM_line:
            self.exper = class_name(exper_name, log_sigma_p)
        else:
            self.exper = class_name(exper_name, scattering_type, mPhi, quenching)

    def compute_data(self, mx, fp, fn, delta, mx_range, vmin_range, initial_energy_bin,
                     logeta_guess, HALO_DEP, MAKE_CROSSES, MAKE_LIMITS, EHI_METHOD,
                     vmin_EHIBand_range, logeta_EHIBand_percent_range, steepness,
                     confidence_levels,
                     vmin_index_list, logeta_index_range, extra_tail):
        """(Re-)compute the data.
        """
        
        output_file = self.output_file_no_extension + "_temp.dat"
        f_handle = open(output_file, 'w')   # clear the file first
        f_handle.close()

        if HALO_DEP:
            (mx_min, mx_max, num_steps) = mx_range
            upper_limit = self.exper.UpperLimit(fp, fn, delta, mx_min, mx_max, num_steps,
                                                output_file)
        else:
            (vmin_min, vmin_max, vmin_step) = vmin_range
            if self.exper_name != "CDMSSi2012" or not np.any(EHI_METHOD):
                if MAKE_LIMITS:
                    upper_limit = \
                        self.exper.UpperLimit(mx, fp, fn, delta, vmin_min, vmin_max,
                                              vmin_step, output_file,
                                              initial_energy_bin=initial_energy_bin)
                elif MAKE_CROSSES and not MAKE_LIMITS:
                    output_file = output_file.replace("UpperLimit", "BinResponseBoxlike")
                    crosses = \
                        self.exper.UpperLimit(mx, fp, fn, delta, vmin_min, vmin_max,
                                              vmin_step, output_file,
                                              initial_energy_bin=initial_energy_bin)

            elif np.any(EHI_METHOD) and self.exper_name == "CDMSSi2012":
                if EHI_METHOD.ResponseTables:
                    self.exper.ResponseTables(vmin_min, vmin_max, vmin_step, mx, fp, fn,
                                              delta, self.output_file_no_extension)
                if EHI_METHOD.OptimalLikelihood:
                    self.exper.OptimalLikelihood(self.output_file_no_extension,
                                                 logeta_guess)
                if EHI_METHOD.ImportOptimalLikelihood:
                    self.exper.ImportResponseTables(self.output_file_no_extension,
                                                    plot=True)
                    self.exper.ImportOptimalLikelihood(self.output_file_no_extension,
                                                       plot=True)
                    self.exper.PlotOptimum()
                if EHI_METHOD.ConstrainedOptimalLikelihood:
                    # Tests for delta = 0:
                    (vminStar, logetaStar) = (500, -25)
                    # Tests for delta = -50:
#                    (vminStar, logetaStar) = (185.572266287, -19.16840262)
                    self.exper.ImportOptimalLikelihood(self.output_file_no_extension)
                    self.exper.ConstrainedOptimalLikelihood(vminStar, logetaStar,
                                                            plot=True)
                if np.any(EHI_METHOD[4:]):
                    if EHI_METHOD._fields[4] != 'VminLogetaSamplingTable':
                        raise AttributeError("EHI_METHOD's attribute is not as expected.")
                    (vmin_Band_min, vmin_Band_max, vmin_Band_numsteps) = \
                        vmin_EHIBand_range
                    (logeta_percent_minus, logeta_percent_plus, logeta_num_steps) = \
                        logeta_EHIBand_percent_range
                    if steepness is not None:
                        (steepness_vmin, steepness_vmin_center, steepness_logeta) = \
                            steepness
                        print("Steepness:", steepness_vmin, ",",
                              steepness_vmin_center, ",", steepness_logeta)
                        self.exper.VminSamplingList(self.output_file_no_extension,
                                                    self.output_file_no_extension,
                                                    vmin_Band_min, vmin_Band_max,
                                                    vmin_Band_numsteps,
                                                    steepness_vmin, steepness_vmin_center,
                                                    MULTI_EXPER,
                                                    plot=not np.any(EHI_METHOD[5:]))
                        self.exper.VminLogetaSamplingTable(self.output_file_no_extension,
                                                           self.output_file_no_extension,
                                                           logeta_percent_minus,
                                                           logeta_percent_plus,
                                                           logeta_num_steps,
                                                           steepness_logeta,
                                                           plot=not np.any(EHI_METHOD[5:]))
                    else:
                        print("Steepness: Default")
                        self.exper.VminSamplingList(self.output_file_no_extension,
                                                    self.output_file_no_extension,
                                                    vmin_Band_min, vmin_Band_max,
                                                    vmin_Band_numsteps,
                                                    MULTI_EXPER,
                                                    plot=not np.any(EHI_METHOD[5:]))
                        self.exper.VminLogetaSamplingTable(self.output_file_no_extension,
                                                           self.output_file_no_extension,
                                                           logeta_percent_minus,
                                                           logeta_percent_plus,
                                                           logeta_num_steps,
                                                           plot=not np.any(EHI_METHOD[5:]))

                if EHI_METHOD.LogLikelihoodList:
                    print("vmin_EHIBand_range =", vmin_Band_min, vmin_Band_max,
                          vmin_Band_numsteps)
                    print("logeta_EHIBand_percent_range =", logeta_percent_minus,
                          logeta_percent_plus, logeta_num_steps)
                    self.exper.LogLikelihoodList(self.output_file_no_extension,
                                                 extra_tail=extra_tail,
                                                 vmin_index_list=vmin_index_list,
                                                 logeta_index_range=logeta_index_range)
                if EHI_METHOD.ConfidenceBand:
                    self.exper.ImportOptimalLikelihood(self.output_file_no_extension)
                    interpolation_order = 2
                    delta_logL = [chi_squared1(c) for c in confidence_levels]
                    for d_logL in delta_logL:
#                        multiplot = (d_logL == delta_logL[0])
                        self.exper.ConfidenceBand(self.output_file_no_extension, d_logL,
                                                  interpolation_order,
                                                  extra_tail=extra_tail,
                                                  multiplot=False)

        if HALO_DEP or not np.any(EHI_METHOD) and MAKE_LIMITS:
            print("upper_limit = ", upper_limit)
            # print("diff response calls = ", self.exper.count_diffresponse_calls)
            # print("response calls = ", self.exper.count_response_calls)
            output_file = self.output_file_no_extension + ".dat"
            print(output_file)  # write to file
            np.savetxt(output_file, upper_limit)
        if MAKE_CROSSES and not MAKE_LIMITS:
            output_file = self.output_file_no_extension + ".dat"
            output_file = output_file.replace("UpperLimit", "BinResponseBoxlike")
            print(output_file)  # write to file
            np.savetxt(output_file, crosses)

    def make_regions(self, delta, confidence_levels):
        """ Make regions for halo-dependent analysis and experiments with potential DM
        signal.
        """

        output_file = self.output_file_no_extension + ".dat"
        #print(output_file)

        for CL in confidence_levels:
            output_file_regions = self.output_file_no_extension + \
                "_" + str(round(sigma_dev(CL), 2)) + "sigma"
            output_file_lower = output_file_regions + "_lower_limit.dat"
            output_file_upper = output_file_regions + "_upper_limit.dat"
            #print(output_file_lower)

            self.exper.Region(delta, CL, output_file, output_file_lower,
                              output_file_upper)

    def plot_limits(self, exper_name, confidence_levels, HALO_DEP, plot_dots, alpha=1):
        plot_data = PlotData(exper_name, HALO_DEP, plot_close=False)
        if (HALO_DEP and exper_name.split()[0] in BinnedSignal_exper) or \
                (HALO_DEP and exper_name=='CDMSSi2012'):
            PlotData.count[exper_name] = -1
            for index, CL in enumerate(confidence_levels):
                output_file_regions = self.output_file_no_extension + \
                    "_" + str(round(sigma_dev(CL), 2)) + "sigma"
                output_file_lower = output_file_regions + "_lower_limit.dat"
                output_file_upper = output_file_regions + "_upper_limit.dat"
                lower_limit = np.loadtxt(output_file_lower)
                upper_limit = np.loadtxt(output_file_upper)
                plot_data(upper_limit, lower_limit=lower_limit, fill=(index < 2),
                          alpha=alpha, plot_dots=plot_dots, plot_show=False)
                if index < 2:
                    PlotData.count[exper_name] -= 1
        else:
            output_file = self.output_file_no_extension + ".dat"
            upper_limit = np.loadtxt(output_file)
            # print("upper_limit = ", upper_limit)
            plot_data(upper_limit, plot_dots=plot_dots, plot_show=False)

    def plot_crosses(self, exper_name, HALO_DEP, alpha=1):
        plot_data = PlotData(exper_name, HALO_DEP, plot_close=False)
        output_file = self.output_file_no_extension + ".dat"
        output_file = output_file.replace("UpperLimit", "BinResponseBoxlike")
        crosses = np.loadtxt(output_file)
        plot_data.plot_crosses(crosses, alpha=alpha, plot_show=False)

    def plot_EHI_band(self, exper_name, confidence_levels, HALO_DEP, extra_tail,
                      plot_dots):
        self.exper.ImportOptimalLikelihood(self.output_file_no_extension)
#        output_file = self.output_file_no_extension + ".dat"
        interp_kind = 'linear'
        plot_limits = PlotData(exper_name, HALO_DEP, plot_close=False)

#        self.exper.PlotSamplingTable(self.output_file_no_extension,
#                                plot_close=False, plot_show=False, plot_optimum=False)
        delta_logL = [chi_squared1(c) for c in confidence_levels]
        print("delta_logL =", delta_logL)
        for d_logL in delta_logL:
            PlotData.count[exper_name] = -1
            self.exper.ImportConfidenceBand(self.output_file_no_extension, d_logL,
                                            extra_tail=extra_tail)
            first_vmin_low = self.exper.vmin_logeta_band_low[0, 0]
            first_vmin_up = self.exper.vmin_logeta_band_up[0, 0]
            last_eta_up = self.exper.vmin_logeta_band_up[-1, 1]
            self.exper.vmin_logeta_band_up = \
                np.vstack(([[first_vmin_low, -10], [first_vmin_up - 5, -10]],
                           self.exper.vmin_logeta_band_up,
                           [[1000, last_eta_up]]))
            last_vmin_low = self.exper.vmin_logeta_band_low[-1, 0]
            last_vmin_up = self.exper.vmin_logeta_band_up[-1, 0]
            self.exper.vmin_logeta_band_low = \
                np.vstack((self.exper.vmin_logeta_band_low,
                           [[last_vmin_low + 5, -40], [last_vmin_up, -40]]))

            plot_limits(self.exper.vmin_logeta_band_up,
                        lower_limit=self.exper.vmin_logeta_band_low, kind=interp_kind,
                        fill=True, alpha=0.2, plot_dots=plot_dots, plot_show=False)

        self.exper.PlotOptimum(ylim_percentage=(1.2, 0.8),
                               color=Color[exper_name + '_EHI'],
                               linewidth=3, plot_close=False,  plot_show=False)

    def __call__(self, exper_name, scattering_type, mPhi, fp, fn, delta,
                 confidence_levels,
                 HALO_DEP, RUN_PROGRAM, MAKE_REGIONS, MAKE_CROSSES, MULTI_EXPER,
                 MAKE_PLOT, EHI_METHOD, MAKE_LIMITS, MULTI_LOGLIKELIST, GENERATE_MC,
                 multiexper_input=None,
                 mx=None, mx_range=None, vmin_range=None, initial_energy_bin=None,
                 vmin_EHIBand_range=None, logeta_EHIBand_percent_range=None,
                 steepness=None, logeta_guess=None,
                 vmin_index_list=None, logeta_index_range=None, log_sigma_p=None,
                 OUTPUT_MAIN_DIR="Output/", filename_tail="", extra_tail="",
                 plot_dots=True, quenching=None):
        """ Main run of the program.
        Input:
            exper_name: string
                Name of experiment.
            scattering_type: string
                Type of scattering. Can be
                - 'SI' (spin-independent)
                - 'SDAV' (spin-independent, axial-vector)
                - 'SDPS' (spin-independent, pseudo-scalar).
            mPhi: float
                Mass of mediator.
            fp and fn: float
                Couplings to proton and neutron.
            delta: float
                DM mass split.
            confidence_levels: list
                List of confidence levels.
            HALO_DEP: bool
                Whether the analysis is halo-dependent or halo-independent.
            RUN_PROGRAM: bool
                Whether the data should be (re-)computed.
            MAKE_REGIONS: bool
                Whether the regions should be (re-)computed in the case of halo-dependent
                analysis and experiments with potential DM signals.
            MAKE_PLOT: bool
                Whether the data should be plotted.
            EHI_Method: ndarray of bools
                Whether each step of the EHI Method is to be performed.
            mx: float, optional
                DM mass, only for halo-independent analysis.
            mx_range: tuple (float, float, int), optional
                (mx_min, mx_max, num_steps) = DM mass range and number or steps,
                only for halo-dependent analysis.
            vmin_range: tuple (float, float, float), optional
                (vmin_min, vmin_max, vmin_step) = vmin range and step size,
                only for halo-independent analysis.
            initial_energy_bin: sequence, optional
                Tuple or list of 2 elements, containing the starting energy bin.
                Only for DAMA combined analysis.
            vmin_EHIBand_range: tuple, optional
                (vmin_Band_min, vmin_Band_max, vmin_Band_numsteps) = vminStar range and
                number of steps, used for calculating the EHI confidence band.
                Only for EHI method.
            logeta_EHIBand_percent_range: tuple, optional
                (logeta_percent_minus, logeta_percent_plus, logeta_num_steps) = logetaStar
                percentage range and number of steps, used for calculating the EHI
                confidence band. The min and max logetaStar are calculated as a given
                percentage above and below the optimum value. Only for EHI method.
            steepness: tuple, optional
                (steepness_vmin, steepness_vmin_center, steepness_logeta) parameters used
                for nonlinear sampling in vminStar and logetaStar. The higher the
                steepnesses the more points are taken close to the steps in the piecewise
                constant best-fit logeta(vmin) function. Only for EHI method.
            logeta_guess: float, optional
                Guessing value of logeta for the best-fit piecewise-constant logeta(vmin)
                function. Only for EHI method.
            vmin_index_list: list, optional
                List of indices in the list of sampling vminStar points for which we
                calculate the optimal likelihood. If not given, the whole list of
                vminStars is used. Only for EHI method.
            logeta_index_range: tuple, optional
                A tuple (index0, index1) between which logetaStar will be considered.
                If not given, then the whole list of logetaStar is used. Only for EHI
                method.
            log_sigma_p: float, optional
                Log base 10 of the total reference cross-section to a single proton.
                Only for halo-independent SHM lines.
            OUTPUT_MAIN_DIR: string, optional
                Name of main output directory.
            filename_tail: string, optional
                Tag to be added to the file name.
            extra_tail: string, optional
                Additional tail to be added to filenames for the EHI confidence band.
            plot_dots: bool, optional
                Whether the plot should show the data points or just the interpolation.
            quenching: float, optional
                quenching factor, needed for experiments that can have multiple options.
        """
        # initialize the experiment class
        self.init_experiment(exper_name, scattering_type, mPhi, delta, HALO_DEP,
                             EHI_METHOD, MAKE_CROSSES, MAKE_LIMITS, log_sigma_p, quenching)

        # get the file name specific to the parameters used for this run
        self.output_file_no_extension = \
            Output_file_name(exper_name, scattering_type, mPhi, mx, fp, fn, delta,
                             HALO_DEP, filename_tail, OUTPUT_MAIN_DIR, quenching)

        # (re-)compute the data
        if RUN_PROGRAM:

            self.compute_data(mx, fp, fn, delta, mx_range, vmin_range, initial_energy_bin,
                              logeta_guess, HALO_DEP, MAKE_CROSSES, MAKE_LIMITS, EHI_METHOD,
                              vmin_EHIBand_range,
                              logeta_EHIBand_percent_range, steepness, confidence_levels,
                              vmin_index_list, logeta_index_range, extra_tail)
            if MAKE_LIMITS and MAKE_CROSSES and exper_name == "CDMSSi2012":
                self.init_experiment(exper_name, scattering_type, mPhi, delta, HALO_DEP,
                                     EHI_METHOD, MAKE_CROSSES, F, log_sigma_p, quenching)
                self.compute_data(mx, fp, fn, delta, mx_range, vmin_range, initial_energy_bin,
                                  logeta_guess, HALO_DEP, MAKE_CROSSES, F, EHI_METHOD,
                                  vmin_EHIBand_range,
                                  logeta_EHIBand_percent_range, steepness, confidence_levels,
                                  vmin_index_list, logeta_index_range, extra_tail)

        # make regions
        if (MAKE_REGIONS and HALO_DEP and exper_name.split()[0] in BinnedSignal_exper) or \
                (MAKE_REGIONS and exper_name in Extended_Like):
            self.make_regions(delta, confidence_levels)

        # make plot
        if MAKE_PLOT and not np.any(EHI_METHOD[:-1]) and MAKE_LIMITS:
            try:
                print(quenching)
                if exper_name.split()[0] in BinnedSignal_exper:
                    alp = transparency.get(quenching, 1)
                elif exper_name == "CDMSSi2012":
                    alp = 0.4
                else:
                    alp = 1.
                self.plot_limits(exper_name, confidence_levels, HALO_DEP, plot_dots,
                                 alpha=alp)
            except np.linalg.linalg.LinAlgError:
                pass

        # make halo-independent crosses
        if MAKE_PLOT and not HALO_DEP and MAKE_CROSSES and \
                (exper_name == "CDMSSi2012" or exper_name.split()[0] == "DAMA2010Na" or
                 exper_name.split()[0] == "DAMA2010I"):
            self.plot_crosses(exper_name, HALO_DEP)

        # make band plot
        if EHI_METHOD.ConfidenceBandPlot and exper_name == "CDMSSi2012":
            self.plot_EHI_band(exper_name, class_name, confidence_levels, HALO_DEP, extra_tail,
                               plot_dots)


class RunProgram_Multiexperiment:
    """ Class implementing the main run of the multiexperiment EHI method.
    """

    def plot_limits(self, exper_name, confidence_levels, HALO_DEP, plot_dots, alpha=1):
        plot_data = PlotData(exper_name, HALO_DEP, plot_close=False)
        if HALO_DEP and exper_name.split()[0] in BinnedSignal_exper:
            PlotData.count[exper_name] = -1
            for index, CL in enumerate(confidence_levels):
                output_file_regions = self.output_file_no_extension + \
                    "_" + str(round(sigma_dev(CL), 2)) + "sigma"
                output_file_lower = output_file_regions + "_lower_limit.dat"
                output_file_upper = output_file_regions + "_upper_limit.dat"
                lower_limit = np.loadtxt(output_file_lower)
                upper_limit = np.loadtxt(output_file_upper)
                plot_data(upper_limit, lower_limit=lower_limit, fill=(index < 2),
                          alpha=alpha, plot_dots=plot_dots, plot_show=False)
                if index < 2:
                    PlotData.count[exper_name] -= 1
        else:
            output_file = self.output_file_no_extension + ".dat"
            upper_limit = np.loadtxt(output_file)
            # print("upper_limit = ", upper_limit)
            plot_data(upper_limit, plot_dots=plot_dots, plot_show=False)

    def plot_crosses(self, exper_name, HALO_DEP, alpha=1):
        plot_data = PlotData(exper_name, HALO_DEP, plot_close=False)
        output_file = self.output_file_no_extension + ".dat"
        output_file = output_file.replace("UpperLimit", "BinResponseBoxlike")
        crosses = np.loadtxt(output_file)
        plot_data.plot_crosses(crosses, alpha=alpha, plot_show=False)

    def plot_EHI_band(self, multiexper_input, class_name, confidence_levels, HALO_DEP, extra_tail, output_file,
                      output_file_CDMS, plot_dots):

        class_name[0].ImportMultiOptimalLikelihood(output_file, output_file_CDMS)
        interp_kind = 'linear'
        plot_limits = PlotData(multiexper_input[0], HALO_DEP, plot_close=False)

#        self.exper.PlotSamplingTable(self.output_file_no_extension,
#                                plot_close=False, plot_show=False, plot_optimum=False)
        delta_logL = [chi_squared1(c) for c in confidence_levels]
        print("delta_logL =", delta_logL)
        for d_logL in delta_logL:
            PlotData.count[multiexper_input[0]] = -1
            class_name[0].ImportConfidenceBand(output_file, d_logL,
                                               extra_tail=extra_tail)
            first_vmin_low = class_name[0].vmin_logeta_band_low[0, 0]
            first_vmin_up = class_name[0].vmin_logeta_band_up[0, 0]
            last_eta_up = class_name[0].vmin_logeta_band_up[-1, 1]
            class_name[0].vmin_logeta_band_up = \
                np.vstack(([[first_vmin_low, -10], [first_vmin_up - 5, -10]],
                           class_name[0].vmin_logeta_band_up,
                           [[1000, last_eta_up]]))
            last_vmin_low = class_name[0].vmin_logeta_band_low[-1, 0]
            last_vmin_up = class_name[0].vmin_logeta_band_up[-1, 0]
            class_name[0].vmin_logeta_band_low = \
                np.vstack((class_name[0].vmin_logeta_band_low,
                           [[last_vmin_low + 5, -40], [last_vmin_up, -40]]))

            plot_limits(class_name[0].vmin_logeta_band_up,
                        lower_limit=class_name[0].vmin_logeta_band_low, kind=interp_kind,
                        fill=True, alpha=0.2, plot_dots=plot_dots, plot_show=False)

        class_name[0].PlotOptimum(ylim_percentage=(1.2, 0.8),
                                  color=Color[multiexper_input[0] + '_EHI'],
                                  linewidth=3, plot_close=False,  plot_show=False)

    def __call__(self, multiexper_input, scattering_type, mPhi, fp, fn, delta,
                 confidence_levels,
                 HALO_DEP, RUN_PROGRAM, MAKE_REGIONS, MULTI_EXPER, GENERATE_MC,
                 MAKE_CROSSES, MAKE_PLOT, EHI_METHOD, MAKE_LIMITS, MULTI_LOGLIKELIST,
                 exper_name=None,
                 mx=None, mx_range=None, vmin_range=None, initial_energy_bin=None,
                 vmin_EHIBand_range=None, logeta_EHIBand_percent_range=None,
                 steepness=None, logeta_guess=None,
                 vmin_index_list=None, logeta_index_range=None, log_sigma_p=None,
                 OUTPUT_MAIN_DIR="Output/", filename_tail="", extra_tail="",
                 plot_dots=True, quenching=None):
        """ Main run of the program.
        Input:
            exper_name: string
                Name of experiment.
            scattering_type: string
                Type of scattering. Can be
                - 'SI' (spin-independent)
                - 'SDAV' (spin-independent, axial-vector)
                - 'SDPS' (spin-independent, pseudo-scalar).
            mPhi: float
                Mass of mediator.
            fp and fn: float
                Couplings to proton and neutron.
            delta: float
                DM mass split.
            confidence_levels: list
                List of confidence levels.
            HALO_DEP: bool
                Whether the analysis is halo-dependent or halo-independent.
            RUN_PROGRAM: bool
                Whether the data should be (re-)computed.
            MAKE_REGIONS: bool
                Whether the regions should be (re-)computed in the case of halo-dependent
                analysis and experiments with potential DM signals.
            MAKE_PLOT: bool
                Whether the data should be plotted.
            EHI_Method: ndarray of bools
                Whether each step of the EHI Method is to be performed.
            mx: float, optional
                DM mass, only for halo-independent analysis.
            mx_range: tuple (float, float, int), optional
                (mx_min, mx_max, num_steps) = DM mass range and number or steps,
                only for halo-dependent analysis.
            vmin_range: tuple (float, float, float), optional
                (vmin_min, vmin_max, vmin_step) = vmin range and step size,
                only for halo-independent analysis.
            initial_energy_bin: sequence, optional
                Tuple or list of 2 elements, containing the starting energy bin.
                Only for DAMA combined analysis.
            vmin_EHIBand_range: tuple, optional
                (vmin_Band_min, vmin_Band_max, vmin_Band_numsteps) = vminStar range and
                number of steps, used for calculating the EHI confidence band.
                Only for EHI method.
            logeta_EHIBand_percent_range: tuple, optional
                (logeta_percent_minus, logeta_percent_plus, logeta_num_steps) = logetaStar
                percentage range and number of steps, used for calculating the EHI
                confidence band. The min and max logetaStar are calculated as a given
                percentage above and below the optimum value. Only for EHI method.
            steepness: tuple, optional
                (steepness_vmin, steepness_vmin_center, steepness_logeta) parameters used
                for nonlinear sampling in vminStar and logetaStar. The higher the
                steepnesses the more points are taken close to the steps in the piecewise
                constant best-fit logeta(vmin) function. Only for EHI method.
            logeta_guess: float, optional
                Guessing value of logeta for the best-fit piecewise-constant logeta(vmin)
                function. Only for EHI method.
            vmin_index_list: list, optional
                List of indices in the list of sampling vminStar points for which we
                calculate the optimal likelihood. If not given, the whole list of
                vminStars is used. Only for EHI method.
            logeta_index_range: tuple, optional
                A tuple (index0, index1) between which logetaStar will be considered.
                If not given, then the whole list of logetaStar is used. Only for EHI
                method.
            log_sigma_p: float, optional
                Log base 10 of the total reference cross-section to a single proton.
                Only for halo-independent SHM lines.
            OUTPUT_MAIN_DIR: string, optional
                Name of main output directory.
            filename_tail: string, optional
                Tag to be added to the file name.
            extra_tail: string, optional
                Additional tail to be added to filenames for the EHI confidence band.
            plot_dots: bool, optional
                Whether the plot should show the data points or just the interpolation.
            quenching: float, optional
                quenching factor, needed for experiments that can have multiple options.
        """
        print('Multiexperiment input list ', multiexper_input)
        num_exper = len(multiexper_input)
        class_name = [None] * num_exper
        pois_main = False
        for x in range(0, num_exper):
            # initialize the experiment class
            print('name = ', multiexper_input[x])
            print('Halo Independent')

            if multiexper_input[x] in Poisson_exper:
                print('PoissonExperiment')
                class_name[x] = PoissonExperiment_HaloIndep(multiexper_input[x], scattering_type, mPhi, quenching)
            elif multiexper_input[x] in GaussianLimit_exper:
                print('GaussianExperiment')
                class_name[x] = GaussianExperiment_HaloIndep(multiexper_input[x], scattering_type, mPhi, quenching)
            elif multiexper_input[x] in Poisson_Like:
                print('Poisson Binned Likelihood')
                class_name[x] = Poisson_Likelihood(multiexper_input[x], scattering_type, mPhi, quenching)
            elif multiexper_input[x] in EHI_Pois:
                print('Poisson EHI')
                pois_main = True
                class_name[x] = Experiment_EHI(multiexper_input[x], scattering_type, mPhi, quenching, pois=True)
            elif multiexper_input[x] == "CDMSSi2012":
                class_name[x] = Experiment_EHI(multiexper_input[x], scattering_type, mPhi, quenching)
            else:
                print("NotImplementedError: This experiment was not implemented!")

        # obtain likelihood
        if RUN_PROGRAM:
            (vmin_min, vmin_max, vmin_step) = vmin_range
            if not pois_main:
                output_file_CDMS = Output_file_name(multiexper_input[0], scattering_type, mPhi, mx, fp, fn, delta,
                                                    HALO_DEP, filename_tail, OUTPUT_MAIN_DIR, quenching=None)
            else:
                output_dir = OutputDirectory(OUTPUT_MAIN_DIR, scattering_type, mPhi, delta)
                output_file_CDMS = output_dir + '/MultiExp_NoEHI_'+ multiexper_input[0] +\
                                   '_mx_{:.2}GeV_fnfp_{:.2f}'.format(mx, fn/fp)
            f_handle = open(output_file_CDMS+"_temp.dat", 'w')   # clear the file first
            f_handle.close()

            if EHI_METHOD.ResponseTables:
                class_name[0].ResponseTables(vmin_min, vmin_max, vmin_step, mx,
                                             fp, fn, delta, output_file_CDMS)


            class_name[0].ImportResponseTables(output_file_CDMS, plot=False)


            output_file = MultiExper_Output_file_name(multiexper_input, scattering_type, mPhi, mx, fp, fn, delta,
                                                      filename_tail, OUTPUT_MAIN_DIR, quenching=None)

            f_handle = open(output_file+"_temp.dat", 'w')   # clear the file first
            f_handle.close()

            if EHI_METHOD.OptimalLikelihood:
                nsteps_bin = 0
                for i in range(1, len(class_name)):
                    nsteps_bin += class_name[i].Nbins

                class_name[0].MultiExperimentOptimalLikelihood(multiexper_input, class_name, mx,
                                                               fp, fn, delta, output_file,
                                                               output_file_CDMS, logeta_guess, nsteps_bin)
                # var_g = np.loadtxt(output_file+"_GloballyOptimalLikelihood.dat")[1:]
                # print(class_name[0]._MinusLogLikelihood(var_g))
                # exit()

            if EHI_METHOD.ImportOptimalLikelihood:
                    class_name[0].ImportResponseTables(output_file_CDMS, plot=False)
                    class_name[0].ImportMultiOptimalLikelihood(output_file, output_file_CDMS, plot=False)
                    if pois_main:
                        kkpt = True
                    else:
                        kkpt = False
                    class_name[0].PlotQ_KKT_Multi(class_name, mx, fp, fn, delta, output_file, kkpt)

            if EHI_METHOD.ConstrainedOptimalLikelihood:
                    # Tests for delta = 0:
                    #print([chi_squared1(c) for c in confidence_levels])
                    
                    (vminStar, logetaStar) = (116.95511, -28.454)
                    # Tests for delta = -50:
#                    (vminStar, logetaStar) = (185.572266287, -19.16840262)
                    class_name[0].ImportMultiOptimalLikelihood(output_file, output_file_CDMS, plot=False)
                    class_name[0].MultiExperConstrainedOptimalLikelihood(vminStar, logetaStar,
                                                                         multiexper_input, class_name,
                                                                         mx, fp, fn, delta, False)

                    exit()
            if np.any(EHI_METHOD[4:]):
                (vmin_Band_min, vmin_Band_max, vmin_Band_numsteps) = \
                    vmin_EHIBand_range
                (logeta_percent_minus, logeta_percent_plus, logeta_num_steps) = \
                    logeta_EHIBand_percent_range
                if steepness is not None:
                        (steepness_vmin, steepness_vmin_center, steepness_logeta) = \
                            steepness
                        print("Steepness:", steepness_vmin, ",",
                              steepness_vmin_center, ",", steepness_logeta)
                        class_name[0].VminSamplingList(output_file, output_file_CDMS,
                                                       vmin_Band_min, vmin_Band_max,
                                                       vmin_Band_numsteps,
                                                       steepness_vmin, steepness_vmin_center,
                                                       MULTI_EXPER,
                                                       plot=False)
                        print('logeta percent minus:', logeta_percent_minus)
                        print('logeta percent plus:', logeta_percent_plus)
                        print('logeta number of steps:', logeta_num_steps)
                        class_name[0].VminLogetaSamplingTable(output_file,
                                                              logeta_percent_minus,
                                                              logeta_percent_plus,
                                                              logeta_num_steps,
                                                              steepness_logeta,
                                                              plot=False)
                else:
                        print("Steepness: Default")
                        class_name[0].VminSamplingList(output_file, output_file_CDMS,
                                                       vmin_Band_min, vmin_Band_max,
                                                       vmin_Band_numsteps,
                                                       MULTI_EXPER,
                                                       plot=not np.any(EHI_METHOD[5:]))

                        class_name[0].VminLogetaSamplingTable(output_file,
                                                              logeta_percent_minus,
                                                              logeta_percent_plus,
                                                              logeta_num_steps,
                                                              plot=not np.any(EHI_METHOD[5:]))

#################### MC SETTINGS ###########

            if GENERATE_MC:
                (vminStar, logetaStar) = (800., -20.)

                MC_directory = "../Output_Band/MC_Files_ContactSI_delta_0/"
                MC_filename = "MC_CDMSSi2012_SuperCDMS_ContactSI_fnfp1_delta0_mx_9GeV_vminStar_" +\
                              str(int(vminStar)) + "_etastar_m" + str(abs(logetaStar)) + ".dat"
                minfunc = np.array([200., 280.3, 438., 677.8, -22., -23.17, -23.17, -23.7])


                for i in range(0, len(class_name)):

                    Nexpected = class_name[i].ExpectedNumEvents(minfunc, mx, fp, fn, delta)

                    Obs_events = class_name[i].Simulate_Events(Nexpected, minfunc, class_name, mx, fp, fn, delta)
                    if i == 0:
                        Tot_list = [Obs_events]
                    else:
                        Tot_list.append(Obs_events)

                LcMaxSuper = class_name[1].Constrained_MC(Obs_events, mx,
                                                          fp, fn, delta, vminStar, logetaStar)
                print('LcMaxSuper: ', LcMaxSuper)
                if Tot_list[0].size > 0:
                    LcMaxCDMS = class_name[0].Constrained_MC_Likelihood(Tot_list[0], vminStar, logetaStar,
                                                                        multiexper_input[0], class_name[0],
                                                                        mx, fp, fn, delta)
                else:
                    LcMaxCDMS = class_name[0]._MinusLogLikelihood(Tot_list[0], vminStar, logetaStar, 0)

                print('LcMaxCDMS: ', LcMaxCDMS)
                if len(Tot_list[0]) > 0:
                    Lcglobal = class_name[0].Constrained_MC_Likelihood(Tot_list[0], vminStar, logetaStar,
                                                                       multiexper_input, class_name,
                                                                       mx, fp, fn, delta)
                else:
                    Lcglobal = LcMaxCDMS + class_name[1]._MinusLogLikelihood(np.array([]), mx, fp, fn, delta,
                                                                             vminStar, logetaStar, vminStar_index=0)

                print('LcGlobal: ', Lcglobal)
                MC_run = np.array([Lcglobal, LcMaxCDMS, LcMaxSuper])
                file_name = MC_directory + MC_filename
                print(MC_run)

                if os.path.exists(file_name):
                    hold = np.loadtxt(file_name)
                    hold = np.vstack((hold, MC_run))
                    np.savetxt(file_name, hold)
                else:
                    np.savetxt(file_name, np.transpose(MC_run))

            if MULTI_LOGLIKELIST:
                (vminStar, logetaStar) = (800., -21.)
                constr_val = class_name[1].Constrained_likelihood(mx, fp, fn, delta, vminStar, logetaStar)
                print('L_constrained_2;max = ', constr_val)

            if EHI_METHOD.LogLikelihoodList:
                    print("vmin_EHIBand_range =", vmin_Band_min, vmin_Band_max, vmin_Band_numsteps)
                    print("logeta_EHIBand_percent_range =", logeta_percent_minus,
                          logeta_percent_plus, logeta_num_steps)
                    class_name[0].ImportMultiOptimalLikelihood(output_file, output_file_CDMS, plot=False)
                    class_name[0].MultiExperLogLikelihoodList(output_file, multiexper_input, class_name,
                                                              mx, fp, fn, delta, scattering_type, mPhi,
                                                              quenching, extra_tail=extra_tail,
                                                              vmin_index_list=vmin_index_list,
                                                              logeta_index_range=logeta_index_range)

            if EHI_METHOD.ConfidenceBand:
                    class_name[0].ImportMultiOptimalLikelihood(output_file, output_file_CDMS)
                    interpolation_order = 2
                    delta_logL = [chi_squared1(c) for c in confidence_levels]
                    if pois_main:
                        if not class_name[0].uniqueBF:
                            delta_logL = [1e-3]
                    for d_logL in delta_logL:
                        class_name[0].ConfidenceBand(output_file, d_logL,
                                                     interpolation_order,
                                                     extra_tail=extra_tail,
                                                     multiplot=False)

        # make band plot
        if EHI_METHOD.ConfidenceBandPlot:
            self.plot_EHI_band(multiexper_input, class_name, confidence_levels, HALO_DEP, extra_tail, output_file,
                               output_file_CDMS, plot_dots)
