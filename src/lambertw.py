"""
Copyright (c) 2015 Andreea Georgescu
Samuel Witte

Created on Monday Sep 21 2015

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

Code taken from http://blogs.mathworks.com/cleve/2013/09/02/the-lambert-w-function/


In case scipy.special.lambertw.py is not correctly installed this can be used.
"""
import math


def lambertw(x, k):
    eps = 0.00000001  # max error allowed

    if k != -1:
        w = 1
    elif k == -1:
        w = -2

    while True:
        ew = math.exp(w)
        f = w * ew - x
        w_new = w - f/((ew * (w + 1.) - (w + 2.) * f/(2. * w + 2.)))
        if abs(w - w_new) <= eps:
            break
        w = w_new
    return w
