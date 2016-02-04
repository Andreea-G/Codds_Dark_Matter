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
import profile
import os   # for speaking
from input_main import *


def main():
    implemented_exper_list = \
        ["SuperCDMS",  # 0
         "LUX2013zero", "LUX2013one", "LUX2013three", "LUX2013five", "LUX2013many",  # 1 - 5
         "SIMPLEModeStage2", "PICASSO", "KIMS2012", "XENON10", "XENON100",  # 6 - 10
         "DAMA2010Na", "DAMA2010I", "DAMA2010Na_TotRateLimit",  # 11 - 13
         "DAMA2010Na DAMA2010I", "DAMA2010I DAMA2010Na",  # 14 - 15
         "CDMSlite2013CoGeNTQ", "CDMSSi2012", "CDMSSiGeArtif", "CDMSSiArtif",  # 16 - 19
         "SHM_eta0", "SHM_eta1"]  # 20 - 21

    # Give input parameters

    EHI_METHOD = {}
    # EHI_METHOD['ResponseTables'] = T
    EHI_METHOD['OptimalLikelihood'] = T
    EHI_METHOD['ImportOptimalLikelihood'] = T
    # EHI_METHOD['ConstrainedOptimalLikelihood'] = T
    # EHI_METHOD['VminLogetaSamplingTable'] = T
    # EHI_METHOD['LogLikelihoodList'] = T
    # EHI_METHOD['ConfidenceBand'] = T
    # EHI_METHOD['ConfidenceBandPlot'] = T

    HALO_DEP = F
    MULTI_EXPER = T  # To be used for multi experiment EHI method
    plot_dots = F
    RUN_PROGRAM = T
    MAKE_LIMITS = F
    MAKE_REGIONS = F
    MAKE_CROSSES = F
    MAKE_PLOT = F
    EXPORT_PLOT = F


    scattering_types = ['SI']  # may be 'SI', 'SDAV', 'SDPS'
    # indices of input_list which can be found in input files
    input_indices = [0]
    # indices of implemented_exper_list
    multiexper_input_indices = [17, 0]
    # indices to be used in multiexperiment EHI anlysis
    exper_indices = []
    OUTPUT_MAIN_DIR = "../Output_Band/"
    filename_tail_list = [""]
    extra_tail = ""

    inp = Input(HALO_DEP, implemented_exper_list, exper_indices=exper_indices,
                input_indices=input_indices,
                multiexper_input_indices = multiexper_input_indices,
                scattering_types=scattering_types,
                RUN_PROGRAM=RUN_PROGRAM, MAKE_REGIONS=MAKE_REGIONS, MULTI_EXPER=MULTI_EXPER,
                MAKE_CROSSES=MAKE_CROSSES, MAKE_LIMITS=MAKE_LIMITS, MAKE_PLOT=MAKE_PLOT,
                EHI_METHOD=EHI_METHOD, OUTPUT_MAIN_DIR=OUTPUT_MAIN_DIR,
                filename_tail_list=filename_tail_list, extra_tail=extra_tail,
                plot_dots=plot_dots)

    # Add or override additional parameters that will be passed to run_program as
    # member variables of the inp class

    inp.initial_energy_bin = [2, 2.5]  # For combined DAMA halo-indep analysis -- Need to choose appropriate mass and Q s.t. energy bins selected appropriately. See arxiv 1502.07682
    # inp.confidence_levels.extend([confidence_level(s) for s in [3, 5]])
    inp.qDAMANa_list = [0.3]

    try:
        plt.close()
        xlim = [0, 1000]
        # ylim = None
        ylim = [-30, -22]
        inp.RunProgram(EXPORT_PLOT=EXPORT_PLOT, xlim=xlim, ylim=ylim)
        if MAKE_PLOT or EHI_METHOD.get('ConfidenceBandPlot', F):
            if not EXPORT_PLOT:
                plt.show()

    finally:
        if inp.RUN_PROGRAM or inp.MAKE_REGIONS:
            #os.system("say 'Finished running program'")
            pass


if __name__ == '__main__':
    main()
    # profile.run("main()")
