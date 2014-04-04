#!/usr/bin/env python

# This code computes the composite SEDs for given SF and AGN templates
# (input files at lines 63-65)
# and plots the correspondent color-color diagrams

import numpy as np
import matplotlib.pyplot as plt
from scipy.interpolate import interp1d
import astropy.constants as const
import os
import os.path

#------------------------------------------------------------------------------
# DEFINITIONS
#------------------------------------------------------------------------------


def get_ab_mag(wave, flux, fnfilter, filter_wave_factor=1.):

    filt = np.genfromtxt(fnfilter, dtype=None, names=['w', 't'])
    filt['w'] = filt['w']*filter_wave_factor  # to angstrom

    filt_interp = np.interp(wave, filt['w'], filt['t'], left=0., right=0.)
    num = np.trapz(flux*filt_interp*wave/const.c.to('angstrom / s'), x=wave)
    den = np.trapz(filt_interp/wave, x=wave)

    return -2.5*np.log10(num/den)-48.6


def make_composite_sed(fn_sfg, fn_AGN, frac_AGN):
    # angstrom, f_lambda
    sfg = np.genfromtxt(fn_sfg, dtype=None, names=['w', 'f'])
    AGN = np.genfromtxt(fn_AGN, dtype=None, names=['w', 'f'])

    # interpolate to common wavelength grid
    wmin = np.min([sfg['w'].min(), AGN['w'].min()])
    wmax = np.max([sfg['w'].max(), AGN['w'].max()])

    wave = np.logspace(np.log10(wmin), np.log10(wmax),
                       num=np.max([sfg['w'].size, AGN['w'].size])*2)

    f_sfg = 10.**np.interp(np.log10(wave), np.log10(sfg['w']),
                           np.log10(sfg['f']), left=-32, right=-32)
    f_AGN = 10.**np.interp(np.log10(wave), np.log10(AGN['w']),
                           np.log10(AGN['f']), left=-32, right=-32)

    l_bol_sfg = np.trapz(f_sfg, x=wave)
    l_bol_AGN = np.trapz(f_AGN, x=wave)

    f_comp = np.zeros((frac_AGN.size, wave.size), dtype=np.float)

    for i in range(frac_AGN.size):
        f_comp[i, :] = (1 - frac_AGN[i])*(f_sfg/l_bol_sfg)
        f_comp[i, :] = f_comp[i, :] + frac_AGN[i]*(f_AGN/l_bol_AGN)

    return wave, f_comp, f_sfg, f_AGN

#------------------------------------------------------------------------------
# MAIN CODE
#------------------------------------------------------------------------------
# read in template SEDs
SW = '/Users/liasartori/Desktop/Astro/dwarf_galaxies/templates/SWIRE_library/'
fn_sfg = SW + 'Ell2_template_norm.sed'  # TO REPLACE
fn_AGN = SW + 'QSO2_template_norm.sed'  # TO REPLACE

name_sfg = 'Ell2'  # TO REPLACE
name_AGN = 'QSO2'  # TO REPLACE

# make composite SEDs
frac_AGN_line = np.logspace(-3, np.log10(0.99), num=100)
wave, flux, fsfg, fAGN = make_composite_sed(fn_sfg, fn_AGN, frac_AGN_line)

frac_AGN = np.array([1., 2., 5., 10., 20., 50., 90.], dtype=np.float) * 1.e-2
wave2, flux2, fsfg2, fAGN2 = make_composite_sed(fn_sfg, fn_AGN, frac_AGN)


# initialize figure
fig = plt.figure(figsize=(18, 11))


# IRAC (Stern et al. 2005) ----------------------------------------------------
# -----------------------------------------------------------------------------

# compute IRAC mag

path_res = '/Users/liasartori/Desktop/Astro/dwarf_galaxies/response_fct/'
path_filt = path_res + 'irac_filters/'
fn_filt = ['ch1.dat', 'ch2.dat', 'ch3.dat', 'ch4.dat']
vega_to_ab = [-2.787, -3.260, -3.753, -4.394]

mag_ch1_vega = np.zeros(flux.shape[0], dtype=np.float)
mag_ch2_vega = np.zeros(flux.shape[0], dtype=np.float)
mag_ch3_vega = np.zeros(flux.shape[0], dtype=np.float)
mag_ch4_vega = np.zeros(flux.shape[0], dtype=np.float)
for i in range(flux.shape[0]):
    mag_ch1_vega[i] = get_ab_mag(wave, flux[i, :],
                                 os.path.join(path_filt, fn_filt[0]))
    mag_ch1_vega[i] = mag_ch1_vega[i] + vega_to_ab[0]
    mag_ch2_vega[i] = get_ab_mag(wave, flux[i, :],
                                 os.path.join(path_filt, fn_filt[1]))
    mag_ch2_vega[i] = mag_ch2_vega[i] + vega_to_ab[1]
    mag_ch3_vega[i] = get_ab_mag(wave, flux[i, :],
                                 os.path.join(path_filt, fn_filt[2]))
    mag_ch3_vega[i] = mag_ch3_vega[i] + vega_to_ab[2]
    mag_ch4_vega[i] = get_ab_mag(wave, flux[i, :],
                                 os.path.join(path_filt, fn_filt[3]))
    mag_ch4_vega[i] = mag_ch4_vega[i] + vega_to_ab[3]

mag_ch1_vega2 = np.zeros(flux2.shape[0], dtype=np.float)
mag_ch2_vega2 = np.zeros(flux2.shape[0], dtype=np.float)
mag_ch3_vega2 = np.zeros(flux2.shape[0], dtype=np.float)
mag_ch4_vega2 = np.zeros(flux2.shape[0], dtype=np.float)
for i in range(flux2.shape[0]):
    mag_ch1_vega2[i] = get_ab_mag(wave2, flux2[i, :],
                                  os.path.join(path_filt, fn_filt[0]))
    mag_ch1_vega2[i] = mag_ch1_vega2[i] + vega_to_ab[0]
    mag_ch2_vega2[i] = get_ab_mag(wave2, flux2[i, :],
                                  os.path.join(path_filt, fn_filt[1]))
    mag_ch2_vega2[i] = mag_ch2_vega2[i] + vega_to_ab[1]
    mag_ch3_vega2[i] = get_ab_mag(wave2, flux2[i, :],
                                  os.path.join(path_filt, fn_filt[2]))
    mag_ch3_vega2[i] = mag_ch3_vega2[i] + vega_to_ab[2]
    mag_ch4_vega2[i] = get_ab_mag(wave2, flux2[i, :],
                                  os.path.join(path_filt, fn_filt[3]))
    mag_ch4_vega2[i] = mag_ch4_vega2[i] + vega_to_ab[3]

# plot color-color diagram
ax = plt.subplot(234)

ax.plot(mag_ch3_vega-mag_ch4_vega, mag_ch1_vega-mag_ch2_vega, '-',
        color='gray')
colorlist = ['Red', 'Orange', 'Sienna', 'Green', 'LightSeaGreen', 'Blue',
             'Fuchsia']
for i in range(7):
    ax.plot(mag_ch3_vega2[i]-mag_ch4_vega2[i],
            mag_ch1_vega2[i]-mag_ch2_vega2[i], 'o', color=colorlist[i])

xmin, xmax = -0.3, 3.3
ymin, ymax = -0.3, 1.5

ax.plot([0.6, 0.6], [0.3, ymax], '--', color='0.3')
ax.plot([0.6, 1.6], [0.3, 0.2*1.6 + 0.18], '--', color='0.3')
ax.plot([1.6, (ymax + 3.5)/2.5], [0.2*1.6 + 0.18,
                                  2.5*((ymax + 3.5)/2.5) - 3.5],
        '--', color='0.3')

for i in range(frac_AGN.size):
    ax.text(mag_ch3_vega2[i]-mag_ch4_vega2[i]+0.1,
            mag_ch1_vega2[i]-mag_ch2_vega2[i],
            '%4.1f%%' % (frac_AGN[i]*100.), fontsize='x-small')

ax.set_xlim(xmin, xmax)
ax.set_ylim(ymin, ymax)
ax.minorticks_on()
ax.set_xlabel('[5.8] - [8.0] (Vega)')
ax.set_ylabel('[3.6] - [4.5] (Vega)')
ax.set_title('Stern et al. (2005)', size=13)

#------------------------------------------------------------------------------


# IRAC+WISE (Stern et al. 2012) -----------------------------------------------
# -----------------------------------------------------------------------------

# compute WISE and IRAC mag
path_filt = '/Users/liasartori/Desktop/Astro/dwarf_galaxies/response_fct/'
fn_filt = ['WISE/280.dat', 'WISE/281.dat', 'irac_filters/ch3.dat',
           'irac_filters/ch4.dat']
vega_to_ab = [-2.683, -3.319, -3.753, -4.394]

mag_ch1_vega = np.zeros(flux.shape[0], dtype=np.float)
mag_ch2_vega = np.zeros(flux.shape[0], dtype=np.float)
mag_ch3_vega = np.zeros(flux.shape[0], dtype=np.float)
mag_ch4_vega = np.zeros(flux.shape[0], dtype=np.float)
for i in range(flux.shape[0]):
    mag_ch1_vega[i] = get_ab_mag(wave, flux[i, :],
                                 os.path.join(path_filt, fn_filt[0]))
    mag_ch1_vega[i] = mag_ch1_vega[i] + vega_to_ab[0]
    mag_ch2_vega[i] = get_ab_mag(wave, flux[i, :],
                                 os.path.join(path_filt, fn_filt[1]))
    mag_ch2_vega[i] = mag_ch2_vega[i] + vega_to_ab[1]
    mag_ch3_vega[i] = get_ab_mag(wave, flux[i, :],
                                 os.path.join(path_filt, fn_filt[2]))
    mag_ch3_vega[i] = mag_ch3_vega[i] + vega_to_ab[2]
    mag_ch4_vega[i] = get_ab_mag(wave, flux[i, :],
                                 os.path.join(path_filt, fn_filt[3]))
    mag_ch4_vega[i] = mag_ch4_vega[i] + vega_to_ab[3]

mag_ch1_vega2 = np.zeros(flux2.shape[0], dtype=np.float)
mag_ch2_vega2 = np.zeros(flux2.shape[0], dtype=np.float)
mag_ch3_vega2 = np.zeros(flux2.shape[0], dtype=np.float)
mag_ch4_vega2 = np.zeros(flux2.shape[0], dtype=np.float)
for i in range(flux2.shape[0]):
    mag_ch1_vega2[i] = get_ab_mag(wave2, flux2[i, :],
                                  os.path.join(path_filt, fn_filt[0]))
    mag_ch1_vega2[i] = mag_ch1_vega2[i] + vega_to_ab[0]
    mag_ch2_vega2[i] = get_ab_mag(wave2, flux2[i, :],
                                  os.path.join(path_filt, fn_filt[1]))
    mag_ch2_vega2[i] = mag_ch2_vega2[i] + vega_to_ab[1]
    mag_ch3_vega2[i] = get_ab_mag(wave2, flux2[i, :],
                                  os.path.join(path_filt, fn_filt[2]))
    mag_ch3_vega2[i] = mag_ch3_vega2[i] + vega_to_ab[2]
    mag_ch4_vega2[i] = get_ab_mag(wave2, flux2[i, :],
                                  os.path.join(path_filt, fn_filt[3]))
    mag_ch4_vega2[i] = mag_ch4_vega2[i] + vega_to_ab[3]

# plot color-color diagram
ax = plt.subplot(235)

ax.plot(mag_ch3_vega-mag_ch4_vega, mag_ch1_vega-mag_ch2_vega, '-',
        color='gray')
colorlist = ['Red', 'Orange', 'Sienna', 'Green', 'LightSeaGreen', 'Blue',
             'Fuchsia']
for i in range(7):
    ax.plot(mag_ch3_vega2[i]-mag_ch4_vega2[i],
            mag_ch1_vega2[i]-mag_ch2_vega2[i], 'o', color=colorlist[i])

ax.plot([-2., 4.], [0.8, 0.8], '--', color='0.3')

for i in range(frac_AGN.size):
    ax.text(mag_ch3_vega2[i]-mag_ch4_vega2[i]+0.1,
            mag_ch1_vega2[i]-mag_ch2_vega2[i], '%4.1f%%' % (frac_AGN[i]*100.),
            fontsize='x-small')

xmin, xmax = -0.3, 3.3
ymin, ymax = -0.3, 1.5
ax.set_xlim(xmin, xmax)
ax.set_ylim(ymin, ymax)
ax.minorticks_on()
ax.set_xlabel('[5.8] - [8.0] (Vega)')
ax.set_ylabel('W1 - W2 (Vega)')
ax.set_title('Stern et al. (2012)', size=13)

#------------------------------------------------------------------------------


# WISE (Jarrett et al. 2011) --------------------------------------------------
# -----------------------------------------------------------------------------

# compute WISE mag
path_filt = '/Users/liasartori/Desktop/Astro/dwarf_galaxies/response_fct/'
fn_filt = ['WISE/280.dat', 'WISE/281.dat', 'WISE/282.dat']
vega_to_ab = [-2.683, -3.319, -5.242]

mag_ch1_vega = np.zeros(flux.shape[0], dtype=np.float)
mag_ch2_vega = np.zeros(flux.shape[0], dtype=np.float)
mag_ch3_vega = np.zeros(flux.shape[0], dtype=np.float)
for i in range(flux.shape[0]):
    mag_ch1_vega[i] = get_ab_mag(wave, flux[i, :],
                                 os.path.join(path_filt, fn_filt[0]))
    mag_ch1_vega[i] = mag_ch1_vega[i] + vega_to_ab[0]
    mag_ch2_vega[i] = get_ab_mag(wave, flux[i, :],
                                 os.path.join(path_filt, fn_filt[1]))
    mag_ch2_vega[i] = mag_ch2_vega[i] + vega_to_ab[1]
    mag_ch3_vega[i] = get_ab_mag(wave, flux[i, :],
                                 os.path.join(path_filt, fn_filt[2]))
    mag_ch3_vega[i] = mag_ch3_vega[i] + vega_to_ab[2]

mag_ch1_vega2 = np.zeros(flux2.shape[0], dtype=np.float)
mag_ch2_vega2 = np.zeros(flux2.shape[0], dtype=np.float)
mag_ch3_vega2 = np.zeros(flux2.shape[0], dtype=np.float)
for i in range(flux2.shape[0]):
    mag_ch1_vega2[i] = get_ab_mag(wave2, flux2[i, :],
                                  os.path.join(path_filt, fn_filt[0]))
    mag_ch1_vega2[i] = mag_ch1_vega2[i] + vega_to_ab[0]
    mag_ch2_vega2[i] = get_ab_mag(wave2, flux2[i, :],
                                  os.path.join(path_filt, fn_filt[1]))
    mag_ch2_vega2[i] = mag_ch2_vega2[i] + vega_to_ab[1]
    mag_ch3_vega2[i] = get_ab_mag(wave2, flux2[i, :],
                                  os.path.join(path_filt, fn_filt[2]))
    mag_ch3_vega2[i] = mag_ch3_vega2[i] + vega_to_ab[2]


# color-color diagram
ax = plt.subplot(236)

ax.plot(mag_ch2_vega-mag_ch3_vega, mag_ch1_vega-mag_ch2_vega, '-',
        color='gray')
colorlist = ['Red', 'Orange', 'Sienna', 'Green', 'LightSeaGreen', 'Blue',
             'Fuchsia']
for i in range(7):
    ax.plot(mag_ch2_vega2[i]-mag_ch3_vega2[i],
            mag_ch1_vega2[i]-mag_ch2_vega2[i], 'o', color=colorlist[i])

ax.plot([2.2, 4.2], [1.7, 1.7], '--', color='0.3')
ax.plot([2.2, 2.2], [0.6, 1.7], '--', color='0.3')
ax.plot([2.2, 4.2], [0.6, 0.8], '--', color='0.3')
ax.plot([4.2, 4.2], [0.8, 1.7], '--', color='0.3')


for i in range(frac_AGN.size):
    ax.text(mag_ch2_vega2[i]-mag_ch3_vega2[i]+0.1,
            mag_ch1_vega2[i]-mag_ch2_vega2[i], '%4.1f%%' % (frac_AGN[i]*100.),
            fontsize='x-small')

xmin, xmax = 0.2, 6.3
ymin, ymax = -0.3, 1.9
ax.set_xlim(xmin, xmax)
ax.set_ylim(ymin, ymax)
ax.minorticks_on()
ax.set_xlabel('W2 - W3 (Vega)')
ax.set_ylabel('W1 - W2 (Vega)')
ax.set_title('Jarrett et al. (2011)', size=13)


# Plot SED templates ----------------------------------------------------------
# -----------------------------------------------------------------------------
# composite SED

ax = plt.subplot(233)

colorlist = ['Red', 'Orange', 'Sienna', 'Green', 'LightSeaGreen', 'Blue',
             'Fuchsia']
for i in range(frac_AGN.size):
    ax.plot(wave2*1e-4, flux2[i, :]*wave2, '-', color=colorlist[i],
            label='%4.1f%%' % (frac_AGN[i]*100.))

path_filt = '/Users/liasartori/Desktop/Astro/dwarf_galaxies/response_fct/'
fn_filt = ['WISE/280.dat', 'WISE/281.dat', 'WISE/282.dat', 'WISE/283.dat',
           'irac_filters/ch1.dat', 'irac_filters/ch2.dat',
           'irac_filters/ch3.dat', 'irac_filters/ch4.dat']

for i in range len(fn_filt):
    fnfilter = os.path.join(path_filt, fn_filt[i])
    filt = np.genfromtxt(fnfilter, dtype=None, names=['w', 't'])
    l_bol = np.trapz(filt['t']*filt['w'], x=filt['w'])
    ax.plot(filt['w']*1e-4, 10**1.5*(filt['t']*filt['w'])/l_bol, '-',
            color='0.3')

ax.set_xscale('log')
ax.set_yscale('log')
ax.set_xlim(0.1, 2000)
ax.set_ylim(1e-5, 1)
ax.set_xlabel(r'$\lambda$ $[\mu m]$', fontsize=16)
ax.set_ylabel(r'$\lambda F_\lambda$ $[norm.]$', fontsize=16)
ax.set_title(name_sfg + ' + ' + name_AGN, size=13)

# Galaxy SED

frac_AGN = np.array([0., 100.], dtype=np.float) * 1.e-2
wave3, flux3, fsfg3, fAGN3 = make_composite_sed(fn_sfg, fn_AGN, frac_AGN)

ax = plt.subplot(231)

ax.plot(wave3*1e-4, flux3[0, :]*wave2, '-', color='black')

path_filt = '/Users/liasartori/Desktop/Astro/dwarf_galaxies/response_fct/'
fn_filt = ['WISE/280.dat', 'WISE/281.dat', 'WISE/282.dat', 'WISE/283.dat',
           'irac_filters/ch1.dat', 'irac_filters/ch2.dat',
           'irac_filters/ch3.dat', 'irac_filters/ch4.dat']

for i in range len(fn_filt):
    fnfilter = os.path.join(path_filt, fn_filt[i])
    filt = np.genfromtxt(fnfilter, dtype=None, names=['w', 't'])
    l_bol = np.trapz(filt['t']*filt['w'], x=filt['w'])
    ax.plot(filt['w']*1e-4, 10**1.5*(filt['t']*filt['w'])/l_bol, '-',
            color='0.3')


ax.set_xscale('log')
ax.set_yscale('log')
ax.set_xlim(0.1, 2000)
ax.set_ylim(1e-5, 1)
ax.set_xlabel(r'$\lambda$ $[\mu m]$', fontsize=16)
ax.set_ylabel(r'$\lambda F_\lambda$ $[norm.]$', fontsize=16)
ax.set_title(name_sfg, size=13)


# AGN SED

ax = plt.subplot(232)

ax.plot(wave3*1e-4, flux3[1, :]*wave2, '-', color='black')

path_filt = '/Users/liasartori/Desktop/Astro/dwarf_galaxies/response_fct/'
fn_filt = ['WISE/280.dat', 'WISE/281.dat', 'WISE/282.dat', 'WISE/283.dat',
           'irac_filters/ch1.dat', 'irac_filters/ch2.dat',
           'irac_filters/ch3.dat', 'irac_filters/ch4.dat']

for i in range len(fn_filt):
    fnfilter = os.path.join(path_filt, fn_filt[i])
    filt = np.genfromtxt(fnfilter, dtype=None, names=['w', 't'])
    l_bol = np.trapz(filt['t']*filt['w'], x=filt['w'])
    ax.plot(filt['w']*1e-4, 10**1.5*(filt['t']*filt['w'])/l_bol, '-',
            color='0.3')


ax.set_xscale('log')
ax.set_yscale('log')
ax.set_xlim(0.1, 2000)
ax.set_ylim(1e-5, 1)
ax.set_xlabel(r'$\lambda$ $[\mu m]$', fontsize=16)
ax.set_ylabel(r'$\lambda F_\lambda$ $[norm.]$', fontsize=16)
ax.set_title(name_AGN, size=13)


# save figures
plt.savefig('plots/color_color/' + name_sfg + '_' + name_AGN,
            bbox_inches='tight')

plt.show()
plt.clf()

#------------------------------------------------------------------------------
#------------------------------------------------------------------------------
