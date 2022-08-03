import matplotlib.pyplot as plt
import numpy as np
from scipy import signal
import scipy.integrate as integrate
import os

import matplotlib.ticker as ticker
from scipy.ndimage import gaussian_filter
from scipy import optimize
from scipy.ndimage import interpolation as intp
from scipy import interpolate as interp
import time
from scipy import special
import sys
from pylab import gca
from EagarTsaiModel import EagarTsai

# This program simulates the heat conduction induced by a moving heat source
# Dependencies: numba, scipy
# Class: Eagar-Tsai implements collective strategy for solving heat source by arranging solutions of the ET equation
## Function =  Graft: Paste new solution onto domain in the correct location
## Function = Forward: Call necessary functions to move heat source for a given time, power and angle
## Function = Check: Determine which boundary conditions are necessary
## Function = Diffuse: Simulate conduction further away from the heat source as a gaussian blur
## Function =  Meltpool: Calculate depth of meltpool
# Class: Solution implements ET solution for a short distance
# Class: Corner solution - Boundary condition where heat source reaches two different walls
# Class: Edge solution - Boundary condition where heat source reaches a single wall
#Certain functions are accelerated with numba and call helper functions (which are indicated with an underscore, e.g. "_solve()"


def zigzag(plotting=False, resolution=10e-6, bc='flux'):
    zz = EagarTsai(resolution=resolution, bc=bc, spacing=200e-6)
    c = 0
    depth = []
    times = []
    times.append(0)
    depth.append(0)
    c = 0
    res_text = str(int(resolution/1e-6))
    file_prefix = "figures/baseline_simulation_figs/zz" + \
        '_' + res_text + '_' + bc + '/'
    os.makedirs(file_prefix, exist_ok=True)
    for iteration in range(21):
        if iteration % 4 == 0:
            for dt in np.arange(0, 1200e-6, 125e-6):

                zz.forward(125e-6, 0)
                depth.append(zz.meltpool())
                times.append(zz.time)
                if plotting:
                    figures = zz.plot()
                    figures[0].savefig(
                        file_prefix + '_x' + '%04d' % c + ".png")
                    figures[1].savefig(
                        file_prefix + '_y' + '%04d' % c + ".png")
                    figures[2].savefig(
                        file_prefix + '_z' + '%04d' % c + ".png")
                c += 1

        if iteration % 4 == 1 or iteration % 4 == 3:
            for dt in np.arange(0, 120e-6, 125e-6):
                zz.forward(125e-6, np.pi/2)
                depth.append(zz.meltpool())
                times.append(zz.time)
                if plotting:
                    figures = zz.plot()
                    figures[0].savefig(
                        file_prefix + '_x' + '%04d' % c + ".png")
                    figures[1].savefig(
                        file_prefix + '_y' + '%04d' % c + ".png")
                    figures[2].savefig(
                        file_prefix + '_z' + '%04d' % c + ".png")
                c += 1
        if iteration % 4 == 2:
            for dt in np.arange(0, 1200e-6, 125e-6):
                zz.forward(125e-6, np.pi)
                depth.append(zz.meltpool())
                times.append(zz.time)
                if plotting:
                    figures = zz.plot()
                    figures[0].savefig(
                        file_prefix + '_x' + '%04d' % c + ".png")
                    figures[1].savefig(
                        file_prefix + '_y' + '%04d' % c + ".png")
                    figures[2].savefig(
                        file_prefix + '_z' + '%04d' % c + ".png")

                c += 1
        plt.close('all')
    if plotting:
        x_oscommand = 'convert ' + file_prefix + \
            '*_x*png -delay 10 ' + file_prefix + 'x.gif'
        y_oscommand = 'convert ' + file_prefix + \
            '*_y*png -delay 10 ' + file_prefix + 'y.gif'
        z_oscommand = 'convert ' + file_prefix + \
            '*_z*png -delay 10 ' + file_prefix + 'z.gif'
        rmcommand = 'rm ' + file_prefix + '*png'
        os.system(x_oscommand)
        os.system(y_oscommand)
        os.system(z_oscommand)
        os.system(rmcommand)

    plt.clf()
    plt.plot(np.array(times)*1e3, np.array(depth)*1e6)
    plt.xlabel(r'Time, $t$ [ms]')
    plt.ylabel(r'Melt Depth, $d$, [$\mu$m]')
    plt.savefig(file_prefix + 'depths.png')
    plt.clf()


def closezigzag(plotting=False, resolution=20e-6, bc='flux'):
    zz = EagarTsai(resolution=resolution, bc=bc, spacing=20e-6)
    c = 0
    depth = []
    times = []
    times.append(0)
    depth.append(0)
    c = 0
    res_text = str(int(resolution/1e-6))
    file_prefix = "figures/baseline_simulation_figs/closezz" + \
        '_' + res_text + '_' + bc + '/'
    os.makedirs(file_prefix, exist_ok=True)
    for iteration in range(21):
        if iteration % 4 == 0:
            for dt in np.arange(0, 1200e-6, 125e-6):

                zz.forward(125e-6, 0, P=145)
                if plotting:
                    figures = zz.plot()
                    figures[0].savefig(
                        file_prefix + '_x' + '%04d' % c + ".png")
                    figures[1].savefig(
                        file_prefix + '_y' + '%04d' % c + ".png")
                    figures[2].savefig(
                        file_prefix + '_z' + '%04d' % c + ".png")
                #plt.show()
                c += 1
                depth.append(zz.meltpool())
                times.append(zz.time)

        if iteration % 4 == 1 or iteration % 4 == 3:
            for dt in np.arange(0, 120e-6, 125e-6):
                zz.forward(125e-6, np.pi/2)
                if plotting:
                    figures = zz.plot()
                    figures[0].savefig(
                        file_prefix + '_x' + '%04d' % c + ".png")
                    figures[1].savefig(
                        file_prefix + '_y' + '%04d' % c + ".png")
                    figures[2].savefig(
                        file_prefix + '_z' + '%04d' % c + ".png")
                c += 1
                depth.append(zz.meltpool())
                times.append(zz.time)
        if iteration % 4 == 2:
            for dt in np.arange(0, 1200e-6, 125e-6):
                zz.forward(125e-6, np.pi)

                if plotting:
                    figures = zz.plot()
                    figures[0].savefig(
                        file_prefix + '_x' + '%04d' % c + ".png")
                    figures[1].savefig(
                        file_prefix + '_y' + '%04d' % c + ".png")
                    figures[2].savefig(
                        file_prefix + '_z' + '%04d' % c + ".png")
                c += 1
                depth.append(zz.meltpool())
                times.append(zz.time)


    if plotting:
        x_oscommand = 'convert ' + file_prefix + \
            '*_x*png -delay 10 ' + file_prefix + 'x.gif'
        y_oscommand = 'convert ' + file_prefix + \
            '*_y*png -delay 10 ' + file_prefix + 'y.gif'
        z_oscommand = 'convert ' + file_prefix + \
            '*_z*png -delay 10 ' + file_prefix + 'z.gif'
        rmcommand = 'rm ' + file_prefix + '*png'
        os.system(x_oscommand)
        os.system(y_oscommand)
        os.system(z_oscommand)
        os.system(rmcommand)
    plt.clf()
    plt.plot(np.array(times)*1e3, np.array(depth)*1e6)
    plt.xlabel(r'Time, $t$ [ms]')
    plt.ylabel(r'Melt Depth, $d$, [$\mu$m]')
    plt.savefig(file_prefix + 'depths.png')
    plt.clf()


def diagonal(plotting=False, resolution=20e-6, bc='flux'):
    diag = EagarTsai(resolution=resolution, bc=bc, spacing=20e-6)
    c = 0
    depth = []
    times = []
    depth.append(0)
    times.append(0)
    V = 0.8
    h = (1000e-6)/7
    res_text = str(int(resolution/1e-6))
    file_prefix = "figures/baseline_simulation_figs/diag" + \
        '_' + res_text + '_' + bc + '/'
    os.makedirs(file_prefix, exist_ok=True)
    for i in range(27):
        if i < 14:
            idx = ((i + 1)/2)
            if i % 4 == 0:
                l = h
                angle = 0
                dtprime = 0
            if i % 4 == 1:
                l = np.sqrt(2*(h**2)*idx**2)
                dtprime = 0
                angle = 3*np.pi/4
            if i % 4 == 2:
                l = h
                dtprime = 0
                angle = np.pi/2
            if i % 4 == 3:

                l = np.sqrt(2*(h**2)*idx**2)
                dtprime = 0
                angle = 7*np.pi/4
        if i >= 14:
            idx = ((27-i)/2)
            if i % 4 == 0:
                l = h
                dtprime = 0
                angle = np.pi/2

            if i % 4 == 1:
                l = np.sqrt(2*(h**2)*idx**2)
                dtprime = 0
                angle = 3*np.pi/4
            if i % 4 == 2:
                l = h
                dtprime = 0
                angle = 0

            if i % 4 == 3:
                l = np.sqrt(2*(h**2)*idx**2)
                dtprime = 0
                angle = 7*np.pi/4

        for dt in np.arange(100e-6, l/V, 100e-6):
            diag.forward(100e-6, angle)
            dtprime += 100e-6
            times.append(diag.time)
            depth.append(diag.meltpool())
            c += 1
            if plotting:
                figures = diag.plot()
                figures[0].savefig(file_prefix + '_x' + '%04d' % c + ".png")
                figures[1].savefig(file_prefix + '_y' + '%04d' % c + ".png")
                figures[2].savefig(file_prefix + '_z' + '%04d' % c + ".png")
        diag.forward(l/V - dtprime, angle)
        times.append(diag.time)
        depth.append(diag.meltpool())
        c += 1
        if plotting:
            figures = diag.plot()
            figures[0].savefig(file_prefix + '_x' + '%04d' % c + ".png")
            figures[1].savefig(file_prefix + '_y' + '%04d' % c + ".png")
            figures[2].savefig(file_prefix + '_z' + '%04d' % c + ".png")
    if plotting:
        x_oscommand = 'convert ' + file_prefix + \
            '*_x*png -delay 10 ' + file_prefix + 'x.gif'
        y_oscommand = 'convert ' + file_prefix + \
            '*_y*png -delay 10 ' + file_prefix + 'y.gif'
        z_oscommand = 'convert ' + file_prefix + \
            '*_z*png -delay 10 ' + file_prefix + 'z.gif'
        rmcommand = 'rm ' + file_prefix + '*png'
        os.system(x_oscommand)
        os.system(y_oscommand)
        os.system(z_oscommand)
        os.system(rmcommand)
    plt.clf()
    plt.plot(np.array(times)*1e3, np.array(depth)*1e6)
    plt.xlabel(r'Time, $t$ [ms]')
    plt.ylabel(r'Melt Depth, $d$, [$\mu$m]')
    plt.savefig(file_prefix + 'depths.png')
    plt.clf()

def triangle(plotting=False, resolution=5e-6, bc='temp'):
    tri = EagarTsai(resolution=resolution, bc=bc, spacing=20e-6)
    c = 0
    depth = []
    times = []
    depth.append(0)
    times.append(0)
    res_text = str(int(resolution/1e-6))
    file_prefix = "figures/baseline_simulation_figs/triangle" + \
        '_' + res_text + '_' + bc + '/'
    os.makedirs(file_prefix, exist_ok=True)
    for i in range(12):
        plt.close('all')
        V = 0.8
        idx = i - 1
        if idx < 0:
            idx = 0
        h = 750e-6 - idx*70e-6
        dtprime = 0
        angle = (i % 3)*2*np.pi/3
        step = 50e-6
        P = 200
        if i == 11:
            P = 0
        for dt in np.arange(step, h/V, step):
            tri.forward(step, angle, V=0.8, P=P)
            depth.append(tri.meltpool())
            times.append(tri.time)
            dtprime += step
            c += 1
            if plotting:
                figures = tri.plot()
                figures[0].savefig(file_prefix + '_x' + '%04d' % c + ".png")
                figures[1].savefig(file_prefix + '_y' + '%04d' % c + ".png")
                figures[2].savefig(file_prefix + '_z' + '%04d' % c + ".png")
        tri.forward(h/V - dtprime, angle, P=P)
        depth.append(tri.meltpool())
        times.append(tri.time)
        c += 1
        if plotting:
            figures = tri.plot()
            figures[0].savefig(file_prefix + '_x' + '%04d' % c + ".png")
            figures[1].savefig(file_prefix + '_y' + '%04d' % c + ".png")
            figures[2].savefig(file_prefix + '_z' + '%04d' % c + ".png")
    if plotting:
        x_oscommand = 'convert ' + file_prefix + \
            '*_x*png -delay 10 ' + file_prefix + 'x.gif'
        y_oscommand = 'convert ' + file_prefix + \
            '*_y*png -delay 10 ' + file_prefix + 'y.gif'
        z_oscommand = 'convert ' + file_prefix + \
            '*_z*png -delay 10 ' + file_prefix + 'z.gif'
        rmcommand = 'rm ' + file_prefix + '*png'
        os.system(x_oscommand)
        os.system(y_oscommand)
        os.system(z_oscommand)
        os.system(rmcommand)
    plt.clf()

    plt.plot(np.array(times)*1e3, np.array(depth)*1e6)
    plt.xlabel(r'Time, $t$ [ms]')
    plt.ylabel(r'Melt Depth, $d$, [$\mu$m]')
    plt.savefig(file_prefix + 'depths.png')
    plt.clf()

def main():
    # Test cases: zigzag, square, triangle, diagonal
    plotting = False
    zigzag(resolution=20e-6, plotting=plotting, bc='flux')
    diagonal(resolution=20e-6, plotting=plotting, bc='flux')
    triangle(resolution=20e-6, plotting=plotting, bc='flux')
    closezigzag(resolution=20e-6, plotting=plotting, bc='flux')


if __name__ == "__main__":
    main()
