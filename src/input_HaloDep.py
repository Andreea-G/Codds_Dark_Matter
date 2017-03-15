"""
Copyright (c) 2015 Andreea Georgescu

Created on Sat Feb 28 21:41:09 2015

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


def DM_mass_range(exper_name, delta, mPhi=1000., quenching=None):
    """ Range and number of steps for the DM mass.
    Input:
        exper_name: string
            Name of experiment.
        delta: float
            DM mass split.
        mPhi: float, optional
            Mass of mediator.
        quenching: float, optional
            quenching factor, needed for experiments that can have multiple options.
    Returns:
        (vmin_min, vmin_max, vmin_step): tuple (float, float, int)
             DM mass range and number or steps
    """
    if exper_name == "SuperCDMS":
        num_steps = 60
        mx_range_options = {(0, 1000.): (5, 100, num_steps),
                            # mx_range_options = {(0, 1000.): (7, 100, num_steps),
                            (0, 0.): (4, 130, num_steps),
                            (-30, 1000.): (2.3, 100, num_steps),
                            (-30, 0.): (2, 100, num_steps),
                            (-50, 1000.): (1.4, 50, num_steps),
                            (50, 1000.): (20, 100, num_steps),
                            (100, 1000.): (30, 100, num_steps),
                            (-200, 1000.): (.5, 7.5, num_steps),
                            (-225, 1000.): (.43, 5., num_steps),
                            (-500, 1000.): (.2, 1, num_steps),
                            }
    elif "SuperCDMSLessT5" in exper_name:
        num_steps = 60
        mx_range_options = {(0, 1000.): (5, 100, num_steps),
                            # mx_range_options = {(0, 1000.): (7.6, 100, num_steps),
                            (0, 0.): (4, 130, num_steps),
                            (-30, 1000.): (2.3, 100, num_steps),
                            (-30, 0.): (2, 100, num_steps),
                            (-50, 1000.): (1.8, 50, num_steps),
                            (50, 1000.): (20, 100, num_steps),
                            (100, 1000.): (30, 100, num_steps),
                            }
    elif "SuperCDMSLikelihood" in exper_name:
        num_steps = 60
        mx_range_options = {(0, 1000.): (5, 100, num_steps),
                            # mx_range_options = {(0, 1000.): (7.6, 100, num_steps),
                            (0, 0.): (4, 130, num_steps),
                            (-30, 1000.): (2.3, 100, num_steps),
                            (-30, 0.): (2, 100, num_steps),
                            (-50, 1000.): (1.8, 50, num_steps),
                            (50, 1000.): (20, 100, num_steps),
                            (100, 1000.): (30, 100, num_steps),
                            }
    elif "LUX2013" in exper_name:
        num_steps = 30
        mx_range_options = {(0, 1000.): (5.5, 100, num_steps),
                            # mx_range_options = {(0, 1000.): (7.6, 100, num_steps),
                            (0, 0.): (5.80, 130, num_steps),
                            (-30, 1000.): (3.95, 100, num_steps),
                            (-30, 0.): (3.95, 100, num_steps),
                            (-50, 1000.): (3.197, 50, num_steps),
                            (50, 1000.): (17.66, 100, num_steps),
                            (100, 1000.): (40, 100, num_steps),
                            (100, 0.): (40, 300, num_steps),
                            (50, 1000.): (17.66, 100, num_steps),
                            (-200, 1000.): (1.35, 10., num_steps),
                            (-225, 1000.): (1.25, 10., num_steps),
                            (-500, 1000.): (0.615, 1, num_steps)
                            }
    elif "LUX2016" in exper_name:
        num_steps = 60
        mx_range_options = {(0, 1000.): (3.45, 100, num_steps),
                            # mx_range_options = {(0, 1000.): (7.6, 100, num_steps),
                            (0, 0.): (5.80, 130, num_steps),
                            (-30, 1000.): (3.95, 100, num_steps),
                            (-30, 0.): (3.95, 100, num_steps),
                            (-50, 1000.): (1.5, 50, num_steps),
                            (50, 1000.): (17.66, 100, num_steps),
                            (100, 1000.): (40, 100, num_steps),
                            (100, 0.): (40, 300, num_steps),
                            (50, 1000.): (17.66, 100, num_steps),
                            (-200, 1000.): (0.57, 10., num_steps),
                            (-225, 1000.): (0.5, 10., num_steps),
                            (-500, 1000.): (0.25, 1, num_steps)
                            }
    elif "CDMS_Snolab_GeHV" in exper_name:
        num_steps = 60
        mx_range_options = {(0, 1000.): (0.5, 100., num_steps),
                            # mx_range_options = {(0, 1000.): (7.6, 100, num_steps),
                            (0, 0.): (5.80, 130, num_steps),
                            (-30, 1000.): (3.95, 100, num_steps),
                            (-30, 0.): (3.95, 100, num_steps),
                            (-50, 1000.): (0.2, 0.73, num_steps),
                            (50, 1000.): (17.66, 100, num_steps),
                            (100, 1000.): (40, 100, num_steps),
                            (100, 0.): (40, 300, num_steps),
                            (50, 1000.): (17.66, 100, num_steps),
                            (-200, 1000.): (.05, 0.1295, num_steps),
                            (-225, 1000.): (.05, 0.2, num_steps),
                            (-500, 1000.): (.3, 1, num_steps)
                            }

    elif "CDMS_Snolab_SiHV" in exper_name:
        num_steps = 60
        mx_range_options = {(0, 1000.): (.5, 100., num_steps),
                            # mx_range_options = {(0, 1000.): (7.6, 100, num_steps),
                            (0, 0.): (5.80, 130, num_steps),
                            (-30, 1000.): (3.95, 100, num_steps),
                            (-30, 0.): (3.95, 100, num_steps),
                            (-50, 1000.): (0.2, 0.4, num_steps),
                            (50, 1000.): (17.66, 100, num_steps),
                            (100, 1000.): (40, 100, num_steps),
                            (100, 0.): (40, 300, num_steps),
                            (50, 1000.): (17.66, 100, num_steps),
                            (-200, 1000.): (.1, 0.28, num_steps),
                            (-500, 1000.): (.3, 1, num_steps)
                            }
    elif "CDMSlite2016" in exper_name:
        num_steps = 60
        mx_range_options = {(0, 1000.): (2, 30, num_steps),
                            (0, 0.): (3, 130, num_steps),
                            (-30, 1000.): (1, 100, num_steps),
                            (-30, 0.): (1, 100, num_steps),
                            (-50, 1000.): (1, 50, num_steps),
                            (50, 1000.): (10, 100, num_steps),
                            (100, 1000.): (30, 100, num_steps),
                            (-200, 1000.): (0.4, 3., num_steps),
                            (-225, 1000.): (0.2, 2.7, num_steps),
                            (-500, 1000.): (0.2, 0.95, num_steps)
                                }
    elif "XENON100" in exper_name:
        num_steps = 40
        mx_range_options = {(0, 1000.): (5.5, 100, num_steps),
                            # mx_range_options = {(0, 1000.): (7.6, 100, num_steps),                                                                                                                                          
                            (0, 0.): (5.80, 130, num_steps),
                            (-30, 1000.): (3.95, 100, num_steps),
                            (-30, 0.): (3.95, 100, num_steps),
                            (-50, 1000.): (3., 50, num_steps),
                            (50, 1000.): (17.66, 100, num_steps),
                            (100, 1000.): (40, 100, num_steps),
                            (100, 0.): (40, 300, num_steps),
                            (50, 1000.): (17.66, 100, num_steps),
                            (-200, 1000.): (0.8, 8.5, num_steps),
                            (-225, 1000.): (0.7, 8.2, num_steps),
                            (-500, 1000.): (0.3, 1, num_steps)
                            }
    elif "Xenon1T" in exper_name:
        num_steps = 40
        mx_range_options = {(0, 1000.): (4., 100, num_steps),
                            # mx_range_options = {(0, 1000.): (7.6, 100, num_steps),                                                                                                                                          
                            (0, 0.): (5.80, 130, num_steps),
                            (-30, 1000.): (3.95, 100, num_steps),
                            (-30, 0.): (3.95, 100, num_steps),
                            (-50, 1000.): (3., 50, num_steps),
                            (50, 1000.): (17.66, 100, num_steps),
                            (100, 1000.): (40, 100, num_steps),
                            (100, 0.): (40, 300, num_steps),
                            (50, 1000.): (17.66, 100, num_steps),
                            (-200, 1000.): (0.8, 10., num_steps),
                            (-225, 1000.): (0.5, 10., num_steps),
                            (-500, 1000.): (0.3, 1, num_steps)
                            }

    elif "DarkSideG2" in exper_name:
        num_steps = 40
        mx_range_options = {(0, 1000.): (15.05, 100, num_steps),
                            # mx_range_options = {(0, 1000.): (7.6, 100, num_steps),                                                                                                                                          
                            (0, 0.): (5.80, 130, num_steps),
                            (-30, 1000.): (3.95, 100, num_steps),
                            (-30, 0.): (3.95, 100, num_steps),
                            (-50, 1000.): (10.05, 50, num_steps),
                            (50, 1000.): (17.66, 100, num_steps),
                            (100, 1000.): (40, 100, num_steps),
                            (100, 0.): (40, 300, num_steps),
                            (50, 1000.): (17.66, 100, num_steps),
                            (-200, 1000.): (4.999, 10., num_steps),
                            (-225, 1000.): (4.62, 10., num_steps),
                            (-500, 1000.): (0.3, 1, num_steps)
                            }
    elif "PICO_500" in exper_name:
        num_steps = 100
        mx_range_options = {(0, 1000.): (4., 100, num_steps),
                            # mx_range_options = {(0, 1000.): (7.6, 100, num_steps),                                                                                                                                          
                            (0, 0.): (5.80, 130, num_steps),
                            (-30, 1000.): (3.95, 100, num_steps),
                            (-30, 0.): (3.95, 100, num_steps),
                            (-50, 1000.): (1.31, 50, num_steps),
                            (50, 1000.): (17.66, 100, num_steps),
                            (100, 1000.): (40, 100, num_steps),
                            (100, 0.): (40, 300, num_steps),
                            (50, 1000.): (17.66, 100, num_steps),
                            (-200, 1000.): (.46, 10., num_steps),
                            (-225, 1000.): (.415, 10., num_steps),
                            (-500, 1000.): (0.3, 1, num_steps)
                            }
    elif "PICO_60" in exper_name:
        num_steps = 100
        mx_range_options = {(0, 1000.): (4., 100, num_steps),
                            # mx_range_options = {(0, 1000.): (7.6, 100, num_steps),                                                                                                                                          
                            (0, 0.): (5.80, 130, num_steps),
                            (-30, 1000.): (3.95, 100, num_steps),
                            (-30, 0.): (3.95, 100, num_steps),
                            (-50, 1000.): (1.31, 50, num_steps),
                            (50, 1000.): (17.66, 100, num_steps),
                            (100, 1000.): (40, 100, num_steps),
                            (100, 0.): (40, 300, num_steps),
                            (50, 1000.): (17.66, 100, num_steps),
                            (-200, 1000.): (.46, 10., num_steps),
                            (-225, 1000.): (.415, 10., num_steps),
                            (-500, 1000.): (0.3, 1, num_steps)
                            }
    elif "XENON10" in exper_name:
        num_steps = 40
        mx_range_options = {(0, 1000.): (5., 100, num_steps),
                            # mx_range_options = {(0, 1000.): (7.6, 100, num_steps),                                                                                                                                          
                            (0, 0.): (5.80, 130, num_steps),
                            (-30, 1000.): (3.95, 100, num_steps),
                            (-30, 0.): (3.95, 100, num_steps),
                            (-50, 1000.): (1.8, 50, num_steps),
                            (50, 1000.): (17.66, 100, num_steps),
                            (100, 1000.): (40, 100, num_steps),
                            (100, 0.): (40, 300, num_steps),
                            (50, 1000.): (17.66, 100, num_steps),
                            (-200, 1000.): (0.67, 3.5, num_steps),
                            (-500, 1000.): (.3, 1, num_steps)
                            }

    elif exper_name == "PandaX":
        num_steps = 60
        mx_range_options = {(0, 1000.): (3.75, 100, num_steps),
                            # mx_range_options = {(0, 1000.): (7.6, 100, num_steps),
                            (0, 0.): (5.80, 130, num_steps),
                            (-30, 1000.): (3.95, 100, num_steps),
                            (-30, 0.): (3.95, 100, num_steps),
                            (-50, 1000.): (1.8, 50, num_steps),
                            (50, 1000.): (17.66, 100, num_steps),
                            (100, 1000.): (40, 100, num_steps),
                            (100, 0.): (40, 300, num_steps),
                            (-200, 1000.): (0.75, 10., num_steps),
                            (-225, 1000.): (0.7, 10., num_steps),
                            (-500, 1000.): (0.315, 1, num_steps)
                            }
    elif exper_name == "Darwin":
        num_steps = 60
        mx_range_options = {(0, 1000.): (7.1, 100, num_steps),
                            # mx_range_options = {(0, 1000.): (7.6, 100, num_steps),
                            (0, 0.): (5.80, 130, num_steps),
                            (-30, 1000.): (3.95, 100, num_steps),
                            (-30, 0.): (3.95, 100, num_steps),
                            (-50, 1000.): (4.39, 50, num_steps),
                            (50, 1000.): (17.66, 100, num_steps),
                            (100, 1000.): (40, 100, num_steps),
                            (100, 0.): (40, 300, num_steps),
                            (-200, 1000.): (2.05, 10., num_steps),
                            (-225, 1000.): (1.89, 10., num_steps),
                            (-500, 1000.): (0.315, 1, num_steps)
                            }
    elif exper_name == "LZ":
        num_steps = 60
        mx_range_options = {(0, 1000.): (3.25, 100, num_steps),
                            (0, 0.): (5.80, 130, num_steps),
                            (-30, 1000.): (3.95, 100, num_steps),
                            (-30, 0.): (3.95, 100, num_steps),
                            (-50, 1000.): (1.46, 50, num_steps),
                            (50, 1000.): (17.66, 100, num_steps),
                            (100, 1000.): (40, 100, num_steps),
                            (100, 0.): (40, 300, num_steps),
                            (50, 1000.): (17.66, 100, num_steps),
                            (-200, 1000.): (0.54, 10., num_steps),
                            (-225, 1000.): (0.49, 10., num_steps),
                            (-500, 1000.): (0.25, 1, num_steps)
                            }
    elif exper_name == "KIMS2012":
        num_steps = 40
        mx_range_options = {(0, 1000.): (10, 100, num_steps),
                            (0, 0.): (10, 130, num_steps),
                            (-30, 1000.): (8, 100, num_steps),
                            (-30, 0.): (10, 100, num_steps),
                            (-50, 1000.): (6, 50, num_steps),
                            (50, 1000.): (17, 100, num_steps),
                            (50, 0.): (17, 100, num_steps),
                            (100, 1000.): (41, 100, num_steps),
                            (100, 0.): (41.08, 300, num_steps),
                            }
    elif exper_name == "SIMPLEModeStage2":
        num_steps = 200
        mx_range_options = {(0, 1000.): (4, 100, num_steps),
                            (0, 0.): (4, 130, num_steps),
                            (-30, 1000.): (2, 100, num_steps),
                            (-30, 0.): (2, 100, num_steps),
                            (-50, 1000.): (1.5, 50, num_steps),
                            (50, 1000.): (18, 100, num_steps),
                            (100, 1000.): (30, 100, num_steps),
                            }
    elif exper_name == "DAMA2010Na":
        num_steps = 60
        if quenching == 0.4:
            mx_range_options = {(0, 1000.): (5, 25, num_steps),
                                # mx_range_options = {(0, 1000.): (8, 15, num_steps),
                                (0, 0.): (5.5, 25, num_steps),
                                (-30, 1000.): (2, 4, num_steps),
                                (-30, 0.): (2, 3, num_steps),
                                (-50, 1000.): (1., 5., num_steps),
                                (-200, 1000.): (0.4, 1.0, num_steps),
                                (-500, 1000.): (0.1, 0.5, num_steps)
                                }
        else:
            mx_range_options = {(0, 1000.): (6, 30, num_steps),
                                # mx_range_options = {(0, 1000.): (10, 25, num_steps),
                                # (0, 0.): (7, 22, num_steps),
                                (0, 0.): (7, 29, num_steps),
                                (-30, 1000.): (3, 5, num_steps),
                                (-30, 0.): (2.5, 4, num_steps),
                                (-50, 1000.): (1.9, 3.5, num_steps),
                                (-200, 1000.): (0.75, 2, num_steps),
                                (-500, 1000.): (0.35, 2, num_steps)
                                }
    elif exper_name == "DAMA2010I":
        num_steps = 80
        if quenching == 0.06:
            mx_range_options = {(0, 1000.): (30, 100, num_steps),
                                # mx_range_options = {(0, 1000.): (55, 85, num_steps),
                                # (0, 0.): (22, 130, num_steps),
                                (0, 0.): (22, 100, num_steps),
                                (-30, 1000.): (25, 100, num_steps),
                                (-30, 0.): (40, 100, num_steps),
                                (-50, 1000.): (25, 50, num_steps),
                                (50, 1000.): (35, 100, num_steps),
                                (50, 0.): (55, 100, num_steps),
                                (100, 1000.): (45, 100, num_steps),
                                (100, 0.): (50, 500, num_steps),
                                # (100, 0.): (50, 90, num_steps),
                                }
        else:
            mx_range_options = {(0, 1000.): (22, 80, num_steps),
                                # mx_range_options = {(0, 1000.): (40, 65, num_steps),
                                # (0, 0.): (29, 90, num_steps),
                                (0, 0.): (22, 100, num_steps),
                                (-30, 1000.): (20, 80, num_steps),
                                (-30, 0.): (30, 100, num_steps),
                                (-50, 1000.): (18, 40, num_steps),
                                (50, 1000.): (30, 100, num_steps),
                                (50, 0.): (40, 100, num_steps),
                                (100, 1000.): (41, 100, num_steps),
                                (100, 0.): (30, 300, num_steps),
                                # (100, 0.): (42, 65, num_steps),
                                }
    elif exper_name == "DAMA2010Na_TotRateLimit":
        num_steps = 60
        # mx_range_options = {(0, 1000.): (3, 15, num_steps),
        mx_range_options = {(0, 1000.): (3, 20, num_steps),
                            (0, 0.): (3, 15, num_steps),
                            (-30, 1000.): (1, 10, num_steps),
                            (-30, 0.): (1, 10, num_steps),
                            (-50, 1000.): (1, 10, num_steps),
                            }

    elif exper_name == "CDMSSi2012":
        num_steps = 60
        # mx_range_options = {(0, 1000.): (3, 15, num_steps),
        mx_range_options = {(0, 1000.): (4, 25, num_steps),
                            (0, 0.): (3, 15, num_steps),
                            (-30, 1000.): (1, 10, num_steps),
                            (-30, 0.): (1, 10, num_steps),
                            (-50, 1000.): (1, 10, num_steps),
                            (-200, 1000.): (0.75, 2, num_steps),
                            (-225, 1000.): (0.5, 2, num_steps),
                            (-500, 1000.): (0.35, 1, num_steps)
                            }
    else:
        num_steps = 60
        if 'SI' in scattering[0]:
            mx_range_options = {(0, 1000.): (5, 30, num_steps),
                                (-50, 1000.): (2, 10, num_steps),
                                (-200, 1000.): (0.4, 2, num_steps),
                                (-500, 1000.): (0.2, 1, num_steps)
                                }
        else:
            mx_range_options = {(0, 1000.): (5, 100, num_steps),
                                (0, 0.): (3, 130, num_steps),
                                (-30, 1000.): (1, 100, num_steps),
                                (-30, 0.): (1, 100, num_steps),
                                (-50, 1000.): (1, 50, num_steps),
                                (50, 1000.): (10, 100, num_steps),
                                (100, 1000.): (30, 100, num_steps),
                                (-200, 1000.): (0.4, 2, num_steps),
                                (-225, 1000.): (0.3, 1.5, num_steps),
                                (-500, 1000.): (0.2, 1, num_steps)
                                }
    return mx_range_options[(delta, mPhi)]


""" List of input values of the form (fn, delta, mPhi).
"""
scattering = 'SI'
if scattering == 'SI':
    input_list = [(1, 0, 1000.), (1, -50, 1000.), (1, -200, 1000.), (1, -500, 1000.),  # 0 - 3
                  (-0.8, 0, 1000.), (-0.8, -50, 1000.), (-0.8, -200, 1000.), (-0.8, -500, 1000.),  # 4 - 7
                  (-0.7, 0, 1000.), (-0.7, -50, 1000.), (-0.7, -200, 1000.), (-0.7, -500, 1000.), # 8 - 11
                  (-0.7,-225, 1000.)] # 12
else:
    input_list = [(-1/16.4, 0, 1000.),  # 0
                  (0, 0, 1000.), (0, -30, 1000.),  (0, -50, 1000.), (0, 0, 0.), (0, -30, 0.),  # 1 - 5
                  (0, 50, 1000.), (0, 100, 1000.), (0, 100, 0.)]  # 6 - 8
