
import matplotlib.pyplot as plt
import numpy as np
from scipy import signal
import scipy.integrate as integrate

import matplotlib.ticker as ticker
from scipy.ndimage import gaussian_filter
from scipy import optimize
from scipy.ndimage import interpolation as intp
from scipy import interpolate as interp
import time
from scipy import special
import sys
from pylab import gca
from skimage import measure

def frame_tick(frame_width=2, tick_width=1.5):
    ax = gca()
    for axis in ['top', 'bottom', 'left', 'right']:
        ax.spines[axis].set_linewidth(frame_width)
    plt.tick_params(direction='in',
                    width=tick_width)


def plot(theta, nrows, ncols, xs, ys, zs):
    figure, axes = plt.subplots(nrows, ncols)
    nrows = 1
    ncols = 3
    xcurrent = np.argmax(theta[:, len(ys)//2, -1])

    pcm0 = axes[0].pcolormesh(
        ys, xs, theta[:, :, -1], shading='gouraud', cmap='jet', vmin=300, vmax=1673)
    pcm1 = axes[1].pcolormesh(zs, xs, theta[:, len(
        ys)//2, :], shading='gouraud', cmap='jet', vmin=300, vmax=1673)
    pcm2 = axes[2].pcolormesh(zs, ys, theta[xcurrent, :, :],
                              shading='gouraud', cmap='jet', vmin=300, vmax=1673)
    pcms = [pcm0, pcm1, pcm2]
    scale_x = 1e-6
    scale_y = 1e-6
    ticks_x = ticker.FuncFormatter(lambda x, pos: '{0:g}'.format(x/scale_x))
    ticks_y = ticker.FuncFormatter(lambda y, pos: '{0:g}'.format(y/scale_y))
    iteration = 0
    for ax, pcm in zip(axes, pcms):
        ax.set_aspect('equal')
        ax.xaxis.set_major_formatter(ticks_x)
        ax.yaxis.set_major_formatter(ticks_y)
        figure.colorbar(pcm, ax=ax)
        if iteration > 0:
            plt.sca(ax)
            plt.xticks([-300e-6, 0])
        iteration += 1

    figure.tight_layout()
# @njit


def _solve(xs, ys, zs, coeff, rxf, rxr, ry, rz, D, V, sigma, dt):

    theta = np.ones((len(xs), len(ys), len(zs)))*300

    for i in range(len(xs)):
        x = xs[i]
        if x > rxr or x < -rxf:
            continue
        for j in range(len(ys)):
            y = ys[j]
            if y > ry or y < -ry:
                continue
            for k in range(len(zs)):
                z = zs[k]
                if z < -rz:
                    continue
                val = 0

                for taubar in np.arange(dt/5000, dt, step=dt/5000):
                    x = xs[i] - V*dt
                    y = ys[j]
                    z = zs[k]
                    start = taubar**(-0.5)/(sigma**2 + 2*D*taubar)
                    exponent = -1*(((x + V*taubar)**2 + y**2) /
                                   (2*sigma**2 + 4*D*taubar) + (z**2)/(4*D*taubar))
                    value = coeff*np.exp(exponent)*start*dt/5000
                    val += value
                theta[i, j, k] += val

    return theta


def _altsolve(xs, ys, zs, phi, coeff, rxf, rxr, ry, rz, D, V, sigma, dt):
    theta = np.ones((len(xs), len(ys), len(zs)))*300
    theta = np.ones((len(xs), len(ys), len(zs)))*300

    integral_result = integrate.fixed_quad(_freefunc, dt/50000, dt, args=(
        coeff, xs[:, None, None, None], ys[None, :, None, None], zs[None, None, :, None], phi, V, D, sigma, dt), n=75)[0]
    theta += integral_result
    return theta


# @njit
def _freefunc(x, coeff, x_coord, y, z, phi, V, D, sigma, dt):
    xp = -V*x*np.cos(phi)
    yp = -V*x*np.sin(phi)
    lmbda = np.sqrt(4*D*x)
    gamma = np.sqrt(2*sigma**2 + lmbda**2)
    start = (4*D*x)**(-3/2)

    termy = sigma*lmbda*np.sqrt(2*np.pi)/(gamma)
    yexp1 = np.exp(-1*((y - yp)**2)/gamma**2)
    termx = termy
    xexp1 = np.exp(-1*((x_coord - xp)**2)/gamma**2)
    yintegral = termy*(yexp1)
    xintegral = termx*xexp1

    zintegral = 2*np.exp(-(z**2)/(4*D*x))
    value = coeff*start*xintegral*yintegral*zintegral
    return value


def _cornersolve(xs, ys, zs, coeff, rxf, rxr, ry, rz, D, V, sigma, dt, dx, dy, phi):
    theta = np.ones((len(xs), len(ys), len(zs)))*300
    theta = np.ones((len(xs), len(ys), len(zs)))*300
    theta += integrate.fixed_quad(_cornerfunc, dt/5000, dt, args=(
        coeff, xs[:, None, None, None], ys[None, :, None, None], zs[None, None, :, None], dx, dy, V, phi, D, sigma, dt), n=50)[0]
    return theta


def _cornerfunc(x, coeff, x_coord, y, z, dx, dy,  V, phi, D, sigma, dt):
    xp = dx - V*x*np.cos(phi)
    yp = dy - V*x*np.sin(phi)
    lmbda = np.sqrt(4*D*x)
    gamma = np.sqrt(2*sigma**2 + lmbda**2)
    start = (4*D*x)**(-3/2)

    term = sigma*lmbda*np.sqrt(np.pi)/(gamma*np.sqrt(2))

    exp1 = np.exp(-1*((x_coord - xp)**2/(gamma**2)))
    erfcarg1 = ((-x_coord/gamma)*(sigma*np.sqrt(2)/lmbda) +
                (-xp/gamma)*(lmbda/(sigma*np.sqrt(2))))
    exp2 = np.exp(-1*((-x_coord-xp)**2/(gamma**2)))
    erfcarg2 = ((x_coord/gamma)*(sigma*np.sqrt(2))/(lmbda) +
                (-xp/gamma)*(lmbda/(sigma*np.sqrt(2))))
    xintegral = term*(exp1*special.erfc(erfcarg1) +
                      exp2*special.erfc(erfcarg2))

    yexp1 = np.exp(-1*((y - yp)**2)/gamma**2)
    yerfcarg1 = ((-y/gamma)*(sigma*np.sqrt(2)/lmbda) +
                 (-yp/gamma)*(lmbda/(sigma*np.sqrt(2))))
    yexp2 = np.exp(-1*((-y-yp)**2/(gamma**2)))
    yerfcarg2 = ((y/gamma)*(sigma*np.sqrt(2))/(lmbda) +
                 (-yp/gamma)*(lmbda/(sigma*np.sqrt(2))))
    yintegral = term*(yexp1*special.erfc(yerfcarg1) +
                      yexp2*special.erfc(yerfcarg2))

    zintegral = 2*np.exp(-(z**2)/(4*D*x))
    value = coeff*start*xintegral*yintegral*zintegral
    return value


def _edgefunc(x, coeff, x_coord, y, z, dx, V, phi, D, sigma, dt):
    xp = dx - V*x*np.cos(phi)
    yp = -V*x*np.sin(phi)
    lmbda = np.sqrt(4*D*x)
    gamma = np.sqrt(2*sigma**2 + lmbda**2)

    start = (4*D*x)**(-3/2)
    termy = sigma*lmbda*np.sqrt(2*np.pi)/(gamma)

    yexp1 = np.exp(-1*((y - yp)**2)/gamma**2)
    term = sigma*lmbda*np.sqrt(np.pi)/(gamma*np.sqrt(2))
    exp1 = np.exp(-1*((x_coord - xp)**2/(gamma**2)))
    erfcarg1 = ((-x_coord/gamma)*(sigma*np.sqrt(2)/lmbda) +
                (-xp/gamma)*(lmbda/(sigma*np.sqrt(2))))
    exp2 = np.exp(-1*((-x_coord - xp)**2/(gamma**2)))
    erfcarg2 = ((x_coord/gamma)*(sigma*np.sqrt(2))/(lmbda) +
                (-xp/gamma)*(lmbda/(sigma*np.sqrt(2))))

    xintegral = term*(exp1*special.erfc(erfcarg1) +
                      exp2*special.erfc(erfcarg2))
    yintegral = termy*(yexp1)
    zintegral = 2*np.exp(-(z**2)/(4*D*x))
    value = coeff*start*xintegral*yintegral*zintegral
    return value


def _edgesolve(xs, ys, zs, coeff, rxf, rxr, ry, rz, D, V, sigma, dt, dx, phi):

    theta = np.ones((len(xs), len(ys), len(zs)))*300
    theta += integrate.fixed_quad(_edgefunc, dt/5000000, dt, args=(
        coeff, xs[:, None, None, None], ys[None, :, None, None], zs[None, None, :, None], dx, V, phi, D, sigma, dt), n=50)[0]

    return theta


#@njit(boundscheck =  True)
def _graft(theta, sol_theta, xs, ys, zs, l_idx, l_idy, l_new_x, l_new_y):
    y_offset = len(ys)//2
    x_offset = len(xs)//2
    x_min = np.argmin(np.abs(xs))
    y_min = np.argmin(np.abs(ys))

    x_roll = -(x_offset) + l_idx + l_new_x
    y_roll = -(y_offset) + l_idy + l_new_y

    theta += np.roll(sol_theta, (x_roll, y_roll, 0), axis=(0, 1, 2)) - 300

    return theta

#@jitclass(spec)


class Solution():
    def __init__(self, dt, T0, phi,  params):
        self.P = params['P']
        self.V = params['V']
        self.sigma = params['sigma']
        self.A = params['A']
        self.rho = params['rho']
        self.cp = params['cp']
        self.k = params['k']
        self.D = self.k/(self.rho*self.cp)
        self.dimstep = params['dimstep']
        self.xs = params['xs']
        self.ys = params['ys']
        self.zs = params['zs']
        self.dt = dt
        self.T0 = T0
        self.a = 4
        self.phi = phi
        self.theta = np.ones(
            (len(self.xs), len(self.ys), len(self.zs)))*self.T0

    def solve(self):

        coeff = self.P*self.A/(2*np.pi*self.rho*self.cp *
                               (self.sigma**2)*(np.pi)**(3/2))
        rxf = self.a*np.sqrt(self.sigma**2 + 2*self.D*self.dt)
        rxr = self.a*np.sqrt(self.sigma**2 + 2*self.D*self.dt) + self.V*self.dt
        ry = self.a*np.sqrt(self.sigma**2 + 2*self.D*self.dt)
        rz = self.a*np.sqrt(2*self.D*self.dt)
        self.theta = _altsolve(self.xs - self.xs[len(self.xs)//2], self.ys - self.ys[len(
            self.ys)//2], self.zs, self.phi, coeff, rxf, rxr, ry, rz, self.D, self.V, self.sigma, self.dt)
        old_idx = len(self.xs)//2
        old_idy = len(self.ys)//2
        return self.theta

    def rotate(self):
        new_theta = np.ones((len(self.xs), len(self.ys), len(self.zs)))*self.T0
        orig_x = np.argmin(np.abs(self.xs))
        orig_y = np.argmin(np.abs(self.ys))
        origin = np.array([orig_x, orig_y])

        new_theta = np.roll(self.theta, (len(self.xs)//2 -
                            origin[0], len(self.ys)//2 - origin[1]), axis=(0, 1))
        rot_theta = intp.rotate(new_theta, angle=np.rad2deg(
            self.phi), reshape=False, cval=self.T0)
        new_theta = np.roll(rot_theta, (-len(self.xs)//2 +
                            origin[0], -len(self.ys)//2 + origin[1]), axis=(0, 1))
        self.theta = new_theta
        return self.theta

    def generate(self):
        return self.solve()

    def plot(self):
        nrows = 1
        ncols = 3
        figure, axes = plt.subplots(nrows, ncols)
        nrows = 1
        ncols = 3
        xcurrent = np.argmax(self.theta[:, len(self.ys)//2, -1])

        pcm0 = axes[0].pcolormesh(self.ys, self.xs, self.theta[:, :, -1],
                                  shading='gouraud', cmap='jet', vmin=300, vmax=1673)
        pcm1 = axes[1].pcolormesh(self.zs, self.xs, self.theta[:, len(
            self.ys)//2, :], shading='gouraud', cmap='jet', vmin=300, vmax=1673)
        pcm2 = axes[2].pcolormesh(self.zs, self.ys, self.theta[xcurrent, :, :],
                                  shading='gouraud', cmap='jet', vmin=300, vmax=1673)
        pcms = [pcm0, pcm1, pcm2]
        scale_x = 1e-6
        scale_y = 1e-6
        ticks_x = ticker.FuncFormatter(
            lambda x, pos: '{0:g}'.format(x/scale_x))
        ticks_y = ticker.FuncFormatter(
            lambda y, pos: '{0:g}'.format(y/scale_y))
        iteration = 0
        for ax, pcm in zip(axes, pcms):
            ax.set_aspect('equal')
            ax.xaxis.set_major_formatter(ticks_x)
            ax.yaxis.set_major_formatter(ticks_y)
            figure.colorbar(pcm, ax=ax)
            if iteration > 0:
                plt.sca(ax)
                plt.xticks([-300e-6, 0])
            iteration += 1
        figure.tight_layout()


#@jitclass(spec)
class CornerSolution():
    def __init__(self, dt, phi, dx, dy, T0, params):
        self.P = params['P']
        self.V = params['V']
        self.sigma = params['sigma']
        self.A = params['A']
        self.rho = params['rho']
        self.cp = params['cp']
        self.k = params['k']
        self.D = self.k/(self.rho*self.cp)
        self.dimstep = params['dimstep']
        self.xs = params['xs']
        self.ys = params['ys']
        self.zs = params['zs']
        self.dt = dt
        self.phi = phi
        self.T0 = T0
        self.a = 4
        self.dx = dx
        self.dy = dy
        self.theta = np.ones(
            (len(self.xs), len(self.ys), len(self.zs)))*self.T0

    def cornersolve(self):

        coeff = self.P*self.A/(2*np.pi*self.rho*self.cp *
                               (self.sigma**2)*(np.pi)**(3/2))
        rxf = self.a*np.sqrt(self.sigma**2 + 2*self.D*self.dt)
        rxr = self.a*np.sqrt(self.sigma**2 + 2*self.D*self.dt) + self.V*self.dt
        ry = self.a*np.sqrt(self.sigma**2 + 2*self.D*self.dt)
        rz = self.a*np.sqrt(2*self.D*self.dt)
        x_offset = self.xs[len(self.xs)//2]
        y_offset = self.ys[len(self.ys)//2]
        self.theta = _cornersolve(self.xs - x_offset, self.ys - y_offset, self.zs, coeff,
                                  rxf, rxr, ry, rz, self.D, self.V, self.sigma, self.dt, self.dx, self.dy, self.phi)
        return self.theta

    def rotate(self):
        new_theta = np.ones((len(self.xs), len(self.ys), len(self.zs)))*self.T0
        orig_x = np.argmin(np.abs(self.xs))
        orig_y = np.argmin(np.abs(self.ys))

        origin = np.array([orig_x, orig_y])

        new_theta = np.roll(self.theta, (len(self.xs)//2 -
                            origin[0], len(self.ys)//2 - origin[1]), axis=(0, 1))
        rot_theta = intp.rotate(new_theta, angle=np.rad2deg(
            self.phi), reshape=False, cval=self.T0)
        new_theta = np.roll(rot_theta, (-len(self.xs)//2 +
                            origin[0], -len(self.ys)//2 + origin[1]), axis=(0, 1))
        self.theta = new_theta
        return self.theta

    def generate(self):
        return self.cornersolve()

    def plot(self):
        nrows = 1
        ncols = 3
        figure, axes = plt.subplots(nrows, ncols)
        nrows = 1
        ncols = 3
        xcurrent = np.argmax(self.theta[:, len(self.ys)//2, -1])

        pcm0 = axes[0].pcolormesh(self.ys, self.xs, self.theta[:, :, -1],
                                  shading='gouraud', cmap='jet', vmin=300, vmax=1673)
        pcm1 = axes[1].pcolormesh(self.zs, self.xs, self.theta[:, len(
            self.ys)//2, :], shading='gouraud', cmap='jet', vmin=300, vmax=1673)
        pcm2 = axes[2].pcolormesh(self.zs, self.ys, self.theta[xcurrent, :, :],
                                  shading='gouraud', cmap='jet', vmin=300, vmax=1673)
        pcms = [pcm0, pcm1, pcm2]
        scale_x = 1e-6
        scale_y = 1e-6
        ticks_x = ticker.FuncFormatter(
            lambda x, pos: '{0:g}'.format(x/scale_x))
        ticks_y = ticker.FuncFormatter(
            lambda y, pos: '{0:g}'.format(y/scale_y))
        iteration = 0
        for ax, pcm in zip(axes, pcms):
            ax.set_aspect('equal')
            ax.xaxis.set_major_formatter(ticks_x)
            ax.yaxis.set_major_formatter(ticks_y)
            figure.colorbar(pcm, ax=ax)
            if iteration > 0:
                plt.sca(ax)
                plt.xticks([-300e-6, 0])
            iteration += 1
        figure.tight_layout()


class EdgeSolution():
    def __init__(self, dt, alpha, dx, T0, params):
        self.P = params['P']
        self.V = params['V']
        self.sigma = params['sigma']
        self.A = params['A']
        self.rho = params['rho']
        self.cp = params['cp']
        self.k = params['k']
        self.D = self.k/(self.rho*self.cp)
        self.dimstep = params['dimstep']
        self.xs = params['xs']
        self.ys = params['ys']
        self.zs = params['zs']
        self.dt = dt
        self.phi = alpha
        self.T0 = T0
        self.a = 4
        self.dx = dx
        self.theta = np.ones(
            (len(self.xs), len(self.ys), len(self.zs)))*self.T0

    def edgesolve(self):
        coeff = self.P*self.A/(2*np.pi*self.rho*self.cp *
                               (self.sigma**2)*(np.pi)**(3/2))
        rxf = self.a*np.sqrt(self.sigma**2 + 2*self.D*self.dt)
        rxr = self.a*np.sqrt(self.sigma**2 + 2*self.D*self.dt) + self.V*self.dt
        ry = self.a*np.sqrt(self.sigma**2 + 2*self.D*self.dt)
        rz = self.a*np.sqrt(2*self.D*self.dt)
        x_offset = self.xs[len(self.xs)//2]
        y_offset = self.ys[len(self.ys)//2]
        self.theta = _edgesolve(self.xs - x_offset, self.ys - y_offset, self.zs, coeff,
                                rxf, rxr, ry, rz, self.D, self.V, self.sigma, self.dt, self.dx, self.phi)
        return self.theta

    def rotate(self):
        new_theta = np.ones((len(self.xs), len(self.ys), len(self.zs)))*self.T0
        orig_x = np.argmin(np.abs(self.xs))
        orig_y = np.argmin(np.abs(self.ys))

        origin = np.array([orig_x, orig_y])

        new_theta = np.roll(self.theta, (len(self.xs)//2 -
                            origin[0], len(self.ys)//2 - origin[1]), axis=(0, 1))
        rot_theta = intp.rotate(new_theta, angle=np.rad2deg(
            self.phi), reshape=False, cval=self.T0)
        new_theta = np.roll(rot_theta, (-len(self.xs)//2 +
                            origin[0], -len(self.ys)//2 + origin[1]), axis=(0, 1))
        self.theta = new_theta
        return self.theta

    def generate(self):
        self.edgesolve()
        return self.theta

    def plot(self):
        nrows = 1
        ncols = 3
        figure, axes = plt.subplots(nrows, ncols)
        nrows = 1
        ncols = 3
        xcurrent = np.argmax(self.theta[:, len(self.ys)//2, -1])

        pcm0 = axes[0].pcolormesh(self.ys, self.xs, self.theta[:, :, -1],
                                  shading='gouraud', cmap='jet', vmin=300, vmax=1673)
        pcm1 = axes[1].pcolormesh(self.zs, self.xs, self.theta[:, len(
            self.ys)//2, :], shading='gouraud', cmap='jet', vmin=300, vmax=1673)
        pcm2 = axes[2].pcolormesh(self.zs, self.ys, self.theta[xcurrent, :, :],
                                  shading='gouraud', cmap='jet', vmin=300, vmax=1673)
        pcms = [pcm0, pcm1, pcm2]
        scale_x = 1e-6
        scale_y = 1e-6
        ticks_x = ticker.FuncFormatter(
            lambda x, pos: '{0:g}'.format(x/scale_x))
        ticks_y = ticker.FuncFormatter(
            lambda y, pos: '{0:g}'.format(y/scale_y))
        iteration = 0
        for ax, pcm in zip(axes, pcms):
            ax.set_aspect('equal')
            ax.xaxis.set_major_formatter(ticks_x)
            ax.yaxis.set_major_formatter(ticks_y)
            figure.colorbar(pcm, ax=ax)
            if iteration > 0:
                plt.sca(ax)
                plt.xticks([-300e-6, 0])
            iteration += 1
        figure.tight_layout()


class EagarTsai():

    "Produce an analytical E-T solution"

    def __init__(self, resolution, V=0.8, bc='flux', spacing=20e-6):
        self.P = 200
        self.V = V
        self.sigma = 13.75e-6
        self.A = 0.3
        self.rho = 7910
        self.cp = 505
        self.k = 21.5
        self.bc = bc
        self.step = 0
        self.dimstep = resolution
        self.time = 0
        b = spacing
        self.xs = np.arange(-b, 1000e-6 + b, step=self.dimstep)
        self.ys = np.arange(-b, 1000e-6 + b, step=self.dimstep)
        self.zs = np.arange(-300e-6, 0 + self.dimstep, step=self.dimstep)

        self.theta = np.ones((len(self.xs), len(self.ys), len(self.zs)))*300
        self.toggle = np.zeros((len(self.xs), len(self.ys)))
        self.D = self.k/(self.rho*self.cp)

        self.location = [0, 0]
        self.location_idx = [
            np.argmin(np.abs(self.xs)), np.argmin(np.abs(self.ys))]
        self.a = 4
        self.times = []
        self.T0 = 300
        self.oldellipse = np.zeros((len(self.xs), len(self.ys)))
        self.store_idx = {}
        self.store = []
        self.visitedx = []
        self.visitedy = []
        self.state = None
        params = {'P': self.P,
                  'V': self.V,
                  'sigma': self.sigma,
                  'A': self.A,
                  'rho': self.rho,
                  'cp': self.cp,
                  'k': self.k,
                  'dimstep': self.dimstep,
                  'xs': self.xs,
                  'ys': self.ys,
                  'zs': self.zs
                  }

    def edgegraft(self, sol, phi, orientation):

        l = sol.V*sol.dt
        l_idx = int(self.location[0]/self.dimstep)
        l_idy = int(self.location[1]/self.dimstep)
        l_x_new = int(self.location[0]/self.dimstep +
                      l*np.cos(phi)/self.dimstep)
        l_y_new = int(self.location[1]/self.dimstep +
                      l*np.sin(phi)/self.dimstep)
        x_offset = len(self.xs)//2
        y_offset = len(self.ys)//2
        y_roll = (l_y_new - y_offset)
        new_theta = np.roll(sol.theta, (-x_offset, y_roll), axis=(0, 1))
        new_theta[-x_offset:, :, :] = 300
        if orientation == 1:

            new_theta = np.flip(new_theta, axis=0)

        if orientation == 2:
            new_theta = np.roll(sol.theta, (-x_offset, 0), axis=(0, 1))
            new_theta[-x_offset:, :, :] = 300
            offset = (len(self.xs) - len(self.ys)) // 2
            midpoint = len(self.ys)//2
            midx = len(self.xs)//2
            rot_theta = np.rot90(new_theta, k=1, axes=(0, 1))
            cut_theta = rot_theta[:, :-offset*2 or None, :]
            pad_theta = np.pad(cut_theta, ((offset, offset),
                               (0, 0), (0, 0)), mode='minimum')
            x_min = np.argmin(np.abs(self.xs))
            new_theta = np.roll(
                pad_theta, (x_min+l_x_new+x_offset, 0, 0), axis=(0, 1, 2))
            new_theta[x_min+l_x_new+x_offset:, :, :] = 300

        if orientation == 3:
            new_theta = np.roll(sol.theta, (-x_offset, 0), axis=(0, 1))
            new_theta[-x_offset:, :, :] = 300
            offset = (len(self.xs) - len(self.ys)) // 2
            midpoint = len(self.ys)//2
            midx = len(self.xs)//2
            rot_theta = np.rot90(new_theta, k=3, axes=(0, 1))
            cut_theta = rot_theta[:, offset*2 or None:, :]
            pad_theta = np.pad(cut_theta, ((offset, offset),
                               (0, 0), (0, 0)), mode='minimum')
            x_min = np.argmin(np.abs(self.xs))
            new_theta = np.roll(
                pad_theta, (x_min+l_x_new+x_offset, 0, 0), axis=(0, 1, 2))
            new_theta[x_min+l_x_new+x_offset:, :, :] = 300

        self.theta += new_theta - 300

        self.location[0] += l*np.cos(phi)
        self.location[1] += l*np.sin(phi)

        self.location_idx[0] = np.argmin(np.abs(self.location[0] - self.xs))
        self.location_idx[1] = np.argmin(np.abs(self.location[1] - self.ys))

        self.visitedx.append(self.location_idx[0])
        self.visitedy.append(self.location_idx[1])

    def cornergraft(self, sol, phi, orientation):
        c = np.where(np.array(orientation) > 0)

        l = sol.V*sol.dt
        l_idx = int(self.location[0]/self.dimstep)
        l_idy = int(self.location[1]/self.dimstep)
        x_offset = len(self.xs)//2
        y_offset = len(self.ys)//2
        new_theta = np.roll(sol.theta, (-x_offset, -y_offset), axis=(0, 1))

        new_theta[-x_offset or None:, :, :] = 300
        new_theta[:, -y_offset or None:, :] = 300
        if np.all(c[0] == [0, 3]):
            new_theta = np.flip(new_theta, axis=1)
        if np.all(c[0] == [1, 2]):
            new_theta = np.flip(new_theta, axis=0)
        if np.all(c[0] == [1, 3]):
            new_theta = np.flip(new_theta, axis=0)
            new_theta = np.flip(new_theta, axis=1)

        self.theta += new_theta - 300

        self.location[0] += l*np.cos(phi)
        self.location[1] += l*np.sin(phi)

        self.location_idx[0] = np.argmin(np.abs(self.location[0] - self.xs))
        self.location_idx[1] = np.argmin(np.abs(self.location[1] - self.ys))

        self.visitedx.append(self.location_idx[0])
        self.visitedy.append(self.location_idx[1])

    def forward(self, dt, phi, V=0.8, P=200):
        self.P = P
        self.V = V
        params = {'P': self.P,
                  'V': V,
                  'sigma': self.sigma,
                  'A': self.A,
                  'rho': self.rho,
                  'cp': self.cp,
                  'k': self.k,
                  'dimstep': self.dimstep,
                  'xs': self.xs,
                  'ys': self.ys,
                  'zs': self.zs
                  }

  #      check if boundary condition is needed:
        corner, edge, ddim, edges, distprime = self.check(dt, phi, V)

        if edge:

            self.state = 'edge'
            c = np.argmax(np.array(edges))
            if c == 0:
                alpha = phi
            if c == 1:
                alpha = np.pi - phi
            if c == 3:
                alpha = phi + np.pi/2
            if c == 2:
                alpha = phi - np.pi/2

            ddim = np.array(ddim)
            dx = ddim[np.where(ddim > -1)[0][0]]
            sol = EdgeSolution(dt, alpha, dx, self.T0, params)
            sol.generate()
            self.diffuse(sol.dt)

            orientation = c

            self.edgegraft(sol, phi, orientation)

        if corner:
            self.state = 'corner'
            side = -1
            ddim = np.array(ddim)
            dx = ddim[np.where(ddim > -1)[0][0]]
            dy = ddim[np.where(ddim > -1)[0][1]]

            c = np.where(np.array(edges) > 0)
            if distprime[1] < distprime[0]:
                side = np.max(c)
            else:
                side = np.min(c)
            assert(side > -1)
            if np.all(c[0] == [0, 2]):
                alpha = phi
            alpha = phi
            mid_y = np.argmin(np.abs(self.ys))
            mid_x = np.argmin(np.abs(self.xs))
            if np.all(c[0] == [0, 3]):
                alpha = 2*np.pi - phi

            if np.all(c[0] == [1, 2]):
                alpha = np.pi - phi

            if np.all(c[0] == [1, 3]):
                alpha = phi + np.pi
            if (len(c[0]) > 2):
                alpha = phi

            sol = CornerSolution(dt, alpha, dx, dy, self.T0, params)
            sol.generate()
            self.diffuse(sol.dt)
            orientation = edges
            self.cornergraft(sol, phi, orientation)

        if not edge and not corner:
            self.state = 'free'
            if (dt, phi, P) in self.store_idx.keys():
                print(P, self.P)
                sol = self.store[self.store_idx[dt, phi, P]]
            else:
                sol = Solution(dt, self.T0, phi, params)
                sol.generate()
                self.store_idx.update({(dt, phi): len(self.store)})
                self.store.append(sol)

            self.diffuse(sol.dt)
            self.graft(sol, phi)

        self.time += dt

    def check(self, dt, phi, V):

        rxf = self.a*np.sqrt(self.sigma**2 + 2*self.D*dt)
        rxr = self.a*np.sqrt(self.sigma**2 + 2*self.D*dt) + self.V*dt
        ry = self.a*np.sqrt(self.sigma**2 + 2*self.D*dt)
        rz = self.a*np.sqrt(2*self.D*dt)
        l_x = V*dt*np.cos(phi)
        l_y = V*dt*np.sin(phi)

        l_idx = V*dt*np.cos(phi)//self.dimstep
        l_idy = V*dt*np.sin(phi)//self.dimstep
        ellipse = self.oldellipse
        corner, edge, ddim, edges, distprime, ellipse = _checkellipse(rxf, rxr, ry, rz, l_x, l_y, l_idx, l_idy, ellipse, self.xs, self.ys, phi, np.round(
            self.location[0], decimals=10),  np.round(self.location[1], decimals=10))

        self.oldellipse = ellipse
        ## Uncomment these lines to plot the laser path through the domain, useful for troubleshooting paths
        # plt.pcolormesh(self.xs, self.ys, ellipse.T, shading = 'gouraud', cmap = 'viridis')
        # #plt.plot([self.location[0] + l_x, self.location[0]], [self.location[1] + l_y, self.location[1]], 'r.-')
        # # if self.state is not None:
        # #     plt.title(self.state + " "+  str(np.round(self.location[0], decimals = 10)) + " " + str(self.location[1]) + " " + str(self.visitedx) + " " + str(self.visitedy))
        # plt.plot(self.xs[self.visitedx], self.ys[self.visitedy], 'r.-')
        # plt.contour(self.xs, self.ys, ellipse.T, [np.max(ellipse) - 1, np.max(ellipse)], colors = 'k')
        # plt.plot([self.location[0]], [self.location[1]], 'g.-')
        # print(np.max(ellipse))
        # plt.gca().set_aspect('equal')
        # plt.pause(0.1)
        # plt.clf()
        return corner, edge, ddim, edges, distprime

    def solve(self, dt):
        "Solves E-T for dt amount of time"
        coeff = self.A*self.P/(self.rho*self.cp*np.sqrt(self.D*4*np.pi**3))
        rxf = self.a*np.sqrt(self.sigma**2 + 2*self.D*dt)
        rxr = self.a*np.sqrt(self.sigma**2 + 2*self.D*dt) + self.V*dt
        ry = self.a*np.sqrt(self.sigma**2 + 2*self.D*dt)
        rz = self.a*np.sqrt(2*self.D*dt)
        params = {'rxf': rxf,
                  'rxr': rxr,
                  'ry': ry,
                  'rz': rz,
                  'coeff': coeff,
                  'D': self.D,
                  'V': self.V,
                  'sigma': self.sigma,
                  'dt': dt}

        return _solve(self.xs, self.ys, self.zs, coeff, rxf, rxr, ry, rz, self.D, self.V, self.sigma, dt)

    def graft(self, sol, phi):
        l = sol.V*sol.dt
        l_new_x = int(np.rint(sol.V*sol.dt*np.cos(phi)/self.dimstep))
        l_new_y = int(np.rint(sol.V*sol.dt*np.sin(phi)/self.dimstep))
        l_idx = int(self.location[0]/self.dimstep)
        l_idy = int(self.location[1]/self.dimstep)
        y = len(self.ys)//2

        self.theta = _graft(self.theta, sol.theta, sol.xs, sol.ys, sol.zs,
                            self.location_idx[0], self.location_idx[1], l_new_x, l_new_y)

        self.location[0] += l*np.cos(phi)
        self.location[1] += l*np.sin(phi)

        self.location_idx[0] = np.argmin(np.abs(self.location[0] - self.xs))
        self.location_idx[1] = np.argmin(np.abs(self.location[1] - self.ys))
        self.visitedx.append(self.location_idx[0])
        self.visitedy.append(self.location_idx[1])

    def reset(self):
        self.theta = np.ones(
            (len(self.xs), len(self.ys), len(self.zs)))*self.T0
        self.location = [0, 0]
        self.location_idx = [
            np.argmin(np.abs(self.xs)), np.argmin(np.abs(self.ys))]
        self.oldellipse = np.zeros((len(self.xs), len(self.ys)))
        self.store_idx = {}
        self.store = []
        self.visitedx = []
        self.visitedy = []
        self.state = None
        self.time = 0

    def func(self, x, h, y, z):
        coeff = self.A*self.P/(self.rho*self.cp*np.sqrt(self.D*4*np.pi**3))
        start = x**(-0.5)/(self.sigma**2 + 2*self.D*x)
        exponent = -1*(((h + self.V*x)**2 + y**2) /
                       (2*self.sigma**2 + 4*self.D*x) + (z**2)/(4*self.D*x))
        value = coeff*np.exp(exponent)*start
        return value

    def get_coords(self):
        return self.xs, self.ys, self.zs
    # Plot cross sections of domain

    def plot(self):
        nrows = 3
        ncols = 1
        figures = []
        axes = []
        for i in range(3):
            fig = plt.figure(dpi=90)
            figures.append(fig)
            axes.append(fig.add_subplot(1, 1, 1))
        xcurrent = np.argmax(self.theta[:, len(self.ys)//2, -1])

        pcm0 = axes[0].pcolormesh(self.xs, self.ys, self.theta[:, :, -1].T,
                                  shading='gouraud', cmap='jet', vmin=300, vmax=1673)
        pcm1 = axes[1].pcolormesh(self.xs, self.zs, self.theta[:, len(
            self.ys)//2, :].T, shading='gouraud', cmap='jet', vmin=300, vmax=1673)
        pcm2 = axes[2].pcolormesh(self.ys, self.zs, self.theta[xcurrent, :, :].T,
                                  shading='gouraud', cmap='jet', vmin=300, vmax=1673)
        pcms = [pcm0, pcm1, pcm2]
        scale_x = 1e-6
        scale_y = 1e-6
        ticks_x = ticker.FuncFormatter(
            lambda x, pos: '{0:g}'.format(x/scale_x))
        ticks_y = ticker.FuncFormatter(
            lambda y, pos: '{0:g}'.format(y/scale_y))
        iteration = 0
        titles = ["X - Y plane", "X - Z plane", "Y - Z plane"]
        axes[0].set_xlabel(r"x [$\mu$m]")
        axes[0].set_ylabel(r"y [$\mu$m]")
        axes[1].set_xlabel(r"x [$\mu$m]")
        axes[1].set_ylabel(r"z [$\mu$m]")
        axes[2].set_xlabel(r"y [$\mu$m]")
        axes[2].set_ylabel(r"z [$\mu$m]")

        for axis, pcm, fig in zip(axes, pcms, figures):
            axis.set_aspect('equal')
            axis.xaxis.set_major_formatter(ticks_x)
            axis.yaxis.set_major_formatter(ticks_y)

            axis.set_title(str(round(self.time*1e6)) + r'[$\mu$s] ' + "Power: " + str(int(np.around(
                self.P))) + "W" + " Velocity: " + str(np.around(self.V, decimals=2)) + r" [m/s]")
            clb = fig.colorbar(pcm, ax=axis)
            clb.ax.set_title(r'T [$K$]')
            iteration += 1
        return figures

    def diffuse(self, dt):
        diffuse_sigma = np.sqrt(2*self.D*dt)
        if dt < 0:
            print("ERROR: dt cannot be negative")
            breakpoint()
        padsize = int((4*diffuse_sigma)//(self.dimstep*2))
        if self.bc == 'temp':
            padsize = int((4*diffuse_sigma)//(self.dimstep*2))
            if padsize == 0:
                padsize = 1
            theta_pad = np.pad(self.theta, ((
                padsize, padsize), (padsize, padsize), (padsize, padsize)), mode='reflect') - 300
            theta_pad_flip = np.copy(theta_pad)
            theta_pad_flip[-padsize:, :, :] = -theta_pad[-padsize:, :, :]
            theta_pad_flip[:padsize, :, :] = -theta_pad[:padsize, :, :]
            theta_pad_flip[:, -padsize:, :] = -theta_pad[:, -padsize:, :]
            theta_pad_flip[:, :padsize, :] = -theta_pad[:, :padsize, :]
            theta_pad_flip[:, :, :padsize] = -theta_pad[:, :, :padsize]
            theta_pad_flip[:, :, -padsize:] = theta_pad[:, :, -padsize:]

            theta_diffuse = gaussian_filter(theta_pad_flip, sigma=diffuse_sigma/self.dimstep)[
                padsize:-padsize, padsize:-padsize, padsize:-padsize] + 300
        if self.bc == 'flux':
            if padsize == 0:
                padsize = 1

            theta_pad = np.pad(self.theta, ((
                padsize, padsize), (padsize, padsize), (padsize, padsize)), mode='reflect') - 300
            theta_pad_flip = np.copy(theta_pad)
            theta_pad_flip[-padsize:, :, :] = theta_pad[-padsize:, :, :]
            theta_pad_flip[:padsize, :, :] = theta_pad[:padsize, :, :]
            theta_pad_flip[:, -padsize:, :] = theta_pad[:, -padsize:, :]
            theta_pad_flip[:, :padsize, :] = theta_pad[:, :padsize, :]
            theta_pad_flip[:, :, :padsize] = -theta_pad[:, :, :padsize]
            theta_pad_flip[:, :, -padsize:] = theta_pad[:, :, -padsize:]

            theta_diffuse = gaussian_filter(theta_pad_flip, sigma=diffuse_sigma/self.dimstep)[
                padsize:-padsize, padsize:-padsize, padsize:-padsize] + 300

        self.theta = theta_diffuse
        return theta_diffuse

    def meltpool(self, calc_length=False, calc_width=False):
        y_center = np.unravel_index(
            np.argmax(self.theta[:, :, -1]), self.theta[:, :, -1].shape)[1]

        if calc_length:
            prop = measure.regionprops(np.array(self.theta[:,:,-1]>1673, dtype = 'int'))
            prop_l = prop[0].major_axis_length*self.dimstep
            length =  prop_l

        if calc_width:
            prop = measure.regionprops(np.array(self.theta[:,:,-1]>1673, dtype = 'int'))
            prop_w = prop[0].minor_axis_length*self.dimstep
            width = prop_w

        depths = []
        for j in range(len(self.ys)):
            for i in range(len(self.xs)):
                if self.theta[i, j, -1] > 1673:
                    g = interp.CubicSpline(self.zs, self.theta[i, j, :] - 1673)
                    root = optimize.brentq(g, self.zs[0], self.zs[-1])

                    depths.append(root)
                    if root < self.toggle[i, j]:
                        self.toggle[i, j] = root

        if len(depths) == 0:
            depth = 0
        else:
            depth = np.min(depths)
        if calc_length and not calc_width:
            return length, depth
        elif calc_width and not calc_length:
            return width, depth
        elif calc_width and calc_length:
            return width, length, depth 
        return depth

    def rotate(self, sol, phi):

        new_theta = np.copy(sol.theta)
        x_offset = len(self.xs)//2
        y_offset = len(self.ys)//2
        origin = np.array([x_offset, y_offset])

        new_theta = np.copy(sol.theta)
        new_theta = np.roll(new_theta, (len(self.xs)//2 -
                            origin[0], len(self.ys)//2 - origin[1]), axis=(0, 1))
        rot_theta = intp.rotate(new_theta, angle=np.rad2deg(
            phi), reshape=False, cval=self.T0)
        new_theta = np.roll(rot_theta, (-len(self.xs)//2 +
                            origin[0], -len(self.ys)//2 + origin[1]), axis=(0, 1))

        return new_theta
#@njit(boundscheck = True)


def _checkellipse(rxf, rxr, ry, rz, l_x, l_y, l_idx, l_idy, ellipse, xs, ys, phi, location_0, location_1):
    corner = False
    edge = False
    xleft = 0
    xright = 0
    yup = 0
    ydown = 0
    dx = -1
    dy = -1
    dxprime = -1
    dyprime = -1
    gamma = phi + np.pi/2
    for i in range(len(xs)):
        x = xs[i]
        for j in range(len(ys)):
            y = ys[j]
            x0 = location_0
            y0 = location_1

            b = ry
            a = rxr
          #  print("RXR", a)
            if (x-x0)*np.sin(gamma) - (y - y0)*np.cos(gamma) > 0:
                a = rxr
            else:
                a = rxf

            if (((x - x0)*np.sin(gamma) - (y - y0)*np.cos(gamma))/a)**2 + (((x - x0)*np.cos(gamma) + (y - y0)*np.sin(gamma))/b)**2 <= 1:
                ellipse[i, j] = 2
                # if i == 0 or i == len(self.xs) - 1 or j == 0 or j == len(self.ys) - 1:
                #     print("BOUNDARY COLLISION")
                if i == 0:
                    xleft = 1
                    dx = x0 + l_x - xs[0]
                    dxprime = x0 - xs[0]
                    # print("left")
                if i == len(xs) - 1:
                    xright = 1
                    dx = xs[-1] - (x0 + l_x)
                    dxprime = xs[-1] - x0
                    # print("right")
                if j == 0:
                    ydown = 1
                    dy = y0 + l_y - ys[0]
                    dyprime = y0 - ys[0]
                    # print("down")
                if j == len(ys) - 1:
                    yup = 1
                    # print("up")
                    dy = ys[-1] - (y0 + l_y)
                    dyprime = ys[-1] - (y0)

    total = xleft + xright + yup + ydown
   # if total == 0:
    #   print("free solution")
    if total == 1:
        edge = True
    if total == 2:
        corner = True

    if total > 2:
        print("Error: More than two edges contacted by laser")
    ddim = [dx, dy]
    edges = [xleft, xright, ydown, yup]
    distprime = [dxprime, dyprime]
    ellipse = ellipse - 1
    return corner, edge, ddim, edges, distprime, ellipse
