# This module includes utility functions that are used to manipulate waveforms.

# %matplotlib inline

# The following line is added in order to avoid:
# _tkinter.TclError: no display name and no $DISPLAY environment variable,
# which occurs when I convert and run this file in a .py script.
# import matplotlib
# matplotlib.use('Agg')

import os, h5py
import numpy as np
from matplotlib import pyplot as plt
import cwt
import utils
from scipy.interpolate import interp1d

import lal
import pycbc.waveform as wfutils
from pycbc.waveform import get_td_waveform, taper_timeseries

pi = np.pi
fig_width_pt = 510.
inches_per_pt = 1.0/72.27               # Convert pt to inches
golden_mean = (np.sqrt(5)-1.0)/2.0         # Aesthetic ratio
fig_width = fig_width_pt*inches_per_pt  # width in inches
fig_height =fig_width*golden_mean       # height in inches
fig_size = (fig_width, fig_height)
# The resulting fig_size is [7.056870070568701, 4.3613855578033265]

fontsize = 16
legendfontsize = 14

params={'text.usetex': True,
        'axes.labelsize': fontsize,
        'font.size': fontsize,
        'legend.fontsize': legendfontsize,
        'xtick.labelsize': fontsize,
        'ytick.labelsize': fontsize,
        'figure.figsize': fig_size,
        'font.weight': 'normal'
       }


# import pylab
# pylab.rcParams.update(params)
# pylab.rcParams['axes.linewidth'] = 1
# pylab.rc('axes', linewidth=1)

def touchbox(ax):
    ax.minorticks_on()
    ax.tick_params('both', length=5, width=1, which='major')
    ax.tick_params('both', length=3.5, width=1, which='minor')
    ax.spines['top'].set_linewidth(1)
    ax.spines['bottom'].set_linewidth(1)
    ax.spines['right'].set_linewidth(1)
    ax.spines['left'].set_linewidth(1)
    return

# plt.style.use('seaborn-darkgrid')

# Total mass of the system in solar masses
# mass=100
mass = 100

# frequency with which the digitized time-series is sampled.
# Time spacing between samples is 1/sample_frequency
sample_frequency = 4096


# Read the input numerical relativity data from the HDF5 file and generate
# the corresponding gravitational wave signal using the parameters in this notebook.
def gen_waveform(numrel_data, iota, phi):
    ## FIXME(Gonghan): It seems that wavelabel is never used.
#     global wavelabel
#     wavelabel=os.PATH.join(savepath, waveformName.split('/')[-1].replace(".h5",""))

    f = h5py.File(numrel_data, 'r')

    # Getting the 2-2 mode polarizations first.
    hp, hc = get_td_waveform(approximant='NR_hdf5',
                                 numrel_data=numrel_data,
                                 mass1=f.attrs['mass1']*mass,
                                 mass2=f.attrs['mass2']*mass,
                                 spin1z=f.attrs['spin1z'],
                                 spin2z=f.attrs['spin2z'],
                                 delta_t=1.0/sample_frequency,
                                 f_lower=30.,
                                 inclination=iota,
                                 coa_phase=phi,
                                 distance=1000,
                                 mode_array=[[2, 2], [2, -2]])
    # Taper waveform for smooth FFTs
    hp = taper_timeseries(hp, tapermethod="TAPER_START")
    hc = taper_timeseries(hc, tapermethod="TAPER_START")
    amp = wfutils.amplitude_from_polarizations(hp, hc)
    peakAmpTime = amp.sample_times[np.argmax(amp)]

    '''mode_array=[[2,2], [2,-2]]'''

    # Getting the full-mode waveform.
    hp, hc = get_td_waveform(approximant='NR_hdf5',
                             numrel_data=numrel_data,
                             mass1=f.attrs['mass1'] * mass,
                             mass2=f.attrs['mass2'] * mass,
                             spin1z=f.attrs['spin1z'],
                             spin2z=f.attrs['spin2z'],
                             delta_t=1.0 / sample_frequency,
                             f_lower=30.,
                             inclination=iota,
                             coa_phase=phi,
                             distance=1000)
    f.close()

    # Taper waveform for smooth FFTs
    hp = taper_timeseries(hp, tapermethod="TAPER_START")
    hc = taper_timeseries(hc, tapermethod="TAPER_START")

    amp = wfutils.amplitude_from_polarizations(hp, hc)
    foft = wfutils.frequency_from_polarizations(hp, hc)

    # Shift time origin to merger

    ## FIXME(Gonghan): Not sure how we want to define zero time.
    sample_times = amp.sample_times - peakAmpTime
#     sample_times = amp.sample_times

    # # Trim the timeseries before the CWT
    # hp_red = hp[:int(sample_frequency)]

    return {"hp": hp, "hc": hc, "amp": amp, "foft": foft, "sample_times": sample_times}


# Decompose the time series data into a time frequency representation, i.e.
# a time-freq map. In this example, we use a continuous wavelet transform (CWT).
def tf_decompose(hp, sample_times, mother_freq, max_scale):
    ## FIXME(Gonghan): fmin and fmax are not used in the script.
    fmin=sample_frequency*mother_freq/max_scale
    fmax=sample_frequency*mother_freq

    cwt_result = cwt.build_cwt(hp.data, sample_times,
							   mother_freq=mother_freq,
							   max_scale=max_scale)

    wfreqs = cwt_result['frequencies']
    wplane = cwt_result['map']

    # Interpolate the spectrogram to a finer frequency grid for smoother plots
    # XXX: worth checking that we don't introduce artifacts here
    interpolant = interp1d(wfreqs, wplane, axis=0)
    wfreqs = np.arange(wfreqs.min(), max_scale)
    wplane = interpolant(wfreqs)
    return {"wfreqs": wfreqs, "wplane": wplane}


# This is a helper function for plot_wavelet_spectrum(...)
# Return plots of the summed-over-time frequency spectrum of a time-frequency map
# over a certain time window.
def plot_t_sum_spectrum(low_time, upp_time, t_range, f_range, tfmap):
    f_spec = utils.get_t_sum_spectrum(low_time, upp_time, t_range, f_range, tfmap)
    fig, ax = plt.subplots(figsize=(fig_width, fig_height))
    ax.plot(f_range, f_spec)
    ax.set_xlabel(r"Frequency (Hz)")
    ax.set_ylabel(r"Amplitude (1)")
    return fig, ax


# Return plots of the summed-over-time frequency spectrum of a time-frequency map
# for a certain waveform and a certain orientation.
# Example: fig, ax = plot_wavelet_spectrum("/waves/GT0577.h5", np.pi, np.pi/2, -0.05, 0.05, 0.5, 512, False)
def plot_wavelet_spectrum(numrel_data, iota, phi, low_time, upp_time,
                          mother_freq, max_scale, in_deg=False):
    wf_data = gen_waveform(numrel_data, iota, phi)
    tf_data = tf_decompose(wf_data["hp"], wf_data["sample_times"],
                           mother_freq, max_scale)
    fig, ax = plot_t_sum_spectrum(low_time, upp_time, wf_data["sample_times"][:-1],
                        tf_data["wfreqs"], tf_data["wplane"])

    if in_deg:
        ax.set_title(("Wavelet spectrum of {}, mother freq = {:.2f}\n" \
                      + r"$\iota={:.2f}^{{\circ}}$, \
                     $\phi={:.2f}^{{\circ}}$, $t\in ({:.3f},{:.3f})$")
                     .format(utils.get_waveform_name(numrel_data),
							 mother_freq,
							 iota * 180 / pi, phi * 180 / pi, low_time, upp_time),
                    fontsize=18)
    else:
        ax.set_title(("Wavelet spectrum of {}, mother freq = {:.2f}\n" \
                      + r"$\iota={:.2f}\pi$, \
                     $\phi={:.2f}\pi$, $t\in ({:.3f},{:.3f})$")
                     .format(utils.get_waveform_name(numrel_data),
							 mother_freq,
							 iota / pi, phi / pi, low_time, upp_time),
                    fontsize=18)
    print type(fig)
    print type(ax)
    return fig, ax


# Return a plot of the Fourier spectrum for a certain waveform.
# Example: frier = plot_fourier_spectrum("/waves/GT0577.h5", np.pi, np.pi/2, 4, 511)
def plot_fourier_spectrum(numrel_data, iota, phi,
                         low_f=None, upp_f=None, in_deg=False):
    wf_data = gen_waveform(numrel_data, iota, phi)
    frier = wf_data["hp"].to_frequencyseries()

#     print len(frier)
#     print type(frier[0])
#     print abs(frier[0])
#     print frier.sample_frequencies

    fig, ax = plt.subplots(figsize=(fig_width, fig_height))
    ax.plot(frier.sample_frequencies, abs(frier))
    ax.set_xlim(low_f, upp_f)
    ax.set_xlabel("Frequency (Hz)")
    ax.set_ylabel("Amplitude (1)")
    ax.get_yaxis().get_offset_text().set_x(-0.085)

    if in_deg:
        ax.set_title(("Fourier spectrum of {}\n" + r"$\iota={:.2f}^{{\circ}}$, \
                     $\phi={:.2f}^{{\circ}}$")
                     .format(utils.get_waveform_name(numrel_data),
							       iota * 180 / PI, phi * 180 / PI),
                    fontsize=18)
    else:
        ax.set_title("Fourier spectrum of {}\n" + r"$\iota={:.2f}\pi$," 
                      r"$\phi={:.2f}\pi$"
                      .format(utils.get_waveform_name(numrel_data),
							         iota / pi, phi / pi),
                      fontsize=18)

    return fig, ax


# Produce plots given in the physical units of the waveform,
# such that time is in seconds, frequency is in Hertz and gravitational
# wave strain is dimensionless.
def plot_physical(numrel_data, iota, phi,
                  sample_times, hp, amp, foft, wfreqs, wplane, max_scale,
                  show_plot=True, in_degree=False, save_plot=True,
                  savepath=None):
    print "I am in plot_physical()!"
    print "mother frequency:", mother_freq, "iota:", iota, "phi:", phi

    plt.close('all')

    fig, ax = plt.subplots(figsize=(fig_width, 2.1*fig_height),
                           sharex=True, nrows=2)


    #
    # Time series
    #
    a = ax[0].plot(sample_times, hp, label=r'$\Re[h(t)]$')
    b = ax[0].plot(sample_times, amp, label=r'$|h(t)|$')
    ax[0].legend()

    #
    # Spectrogram
    #
    p_qstrain = ax[1].pcolormesh(sample_times, wfreqs, (abs(wplane)),
                              rasterized=False,
                             vmin=0,vmax=1,cmap='gnuplot')

    #
    # Frequency series
    #
    # FIXME: why is frequency negative??
    ax[1].plot(sample_times[:-1], -1*foft, label=r'$\arg[h(t)]$',
               color='green', linestyle='-')

    ax[1].legend(loc='upper left')

    # XXX Do not delete this, in case we want a colorbar later
#     cbaxes = fig.add_axes([0.1275, 0.9, 0.77, 0.03])
#     cbar=fig.colorbar(p_qstrain, orientation='horizontal', cax=cbaxes)
#     cbar.ax.xaxis.set_ticks_position('top')
#     cbar.ax.xaxis.set_label_position('top')
#     ax.clabel(c_qstrain, inline=1, fmt="%.1f", fontsize=14)


    for i in xrange(len(ax)):

        ax[i].set_xlim(-0.05,0.05)

        if i==0:
            ax[i].set_ylabel(r'$\textrm{Strain at 1 Gpc}$')
        if i==1:
            ax[i].set_ylim(0,max_scale)
            ax[i].set_xlabel(r'$\textrm{Time (s)}$')
            ax[i].set_ylabel(r'$\textrm{Frequency (Hz)}$')

        # Beautify the lines
        touchbox(ax[i])

    plt.subplots_adjust(hspace=0)

    if save_plot:
        # Determine figure name from numrel file and orientation
        if in_degree:
            figname = os.path.join(savepath,
                                   os.path.basename(numrel_data).replace('.h5', '')
                                   + '_iota-%f' % (iota * 180 / np.pi)
                                   + '_phi-%f' % (phi * 180 / np.pi) + '.png')
        else:
            figname = os.path.join(savepath,
                                   os.path.basename(numrel_data).replace('.h5', '')
                                   + '_iota-%f' % iota + '_phi-%f' % phi + '.png')
        plt.savefig(figname)

    if show_plot:
        plt.show()

    return fig


# Produce plots where we rescale the data such that G=c=1
# and time is in solar masses in seconds, frequency is in units of 1/solar mass,
# and strain amplitude is in solar masses in meters.
def plot_geometric(numrel_data, iota, phi,
                   sample_times, hp, amp, foft, wfreqs, wplane, max_scale,
                   show_plot=True, in_degree=False, save_plot=True,
                   savepath=None):
    print "I am in plot_geometric()!"
    print "mother frequency:", mother_freq, "iota:", iota, "phi:", phi

    mscale_sec=lal.MTSUN_SI*mass
    mscale_mpc=1/ ( lal.MRSUN_SI / ( 1000 * lal.PC_SI * 1.0e6) )

    plt.close('all')
    fig, ax = plt.subplots(figsize=(fig_width, 2.1*fig_height),
                           sharex=True, nrows=2)


    #
    # Time series
    #
    ax[0].plot(sample_times/mscale_sec, hp*mscale_mpc, label=r'$\Re[h(t)]$')
    a = ax[0].plot(sample_times/mscale_sec, amp*mscale_mpc, label=r'$|h(t)|$')
    ax[0].legend()

    #
    # Spectrogram
    #
    p_qstrain = ax[1].pcolormesh(sample_times/mscale_sec, wfreqs*mscale_sec,
                                 (abs(wplane)), cmap='gnuplot',
                                 rasterized=False, vmin=0,vmax=1)

    #
    # Frequency series
    #
    # FIXME: why is frequency negative??
    ax[1].plot(sample_times[:-1]/mscale_sec, -1*foft*mscale_sec,
               label=r'$\arg[h(t)]$', color='g')

    ax[1].legend(loc='upper left')


    for i in xrange(len(ax)):

        ax[i].set_xlim(-0.05/mscale_sec,0.05/mscale_sec)

        if i==0:
            ax[i].set_ylabel(r'$\textrm{Strain}$')
        if i==1:
            ax[i].set_ylim(10*mscale_sec,max_scale*mscale_sec)
            ax[i].set_xlabel(r'$\textrm{Time }(\textrm{M}_{\odot})$')
            ax[i].set_ylabel(r'$\textrm{Frequency }(1/\textrm{M}_{\odot})$')

        # Beautify the lines
        touchbox(ax[i])

    plt.subplots_adjust(hspace=0)

    if save_plot:
        if in_degree:
            figname = os.path.join(savepath,
                                   os.path.basename(numrel_data).replace('.h5', '')
                                   + '_iota-%f' % (iota * 180 / np.pi)
                                   + '_phi-%f' % (phi * 180 / np.pi) + '_GEOM.png')
        else:
            figname = os.path.join(savepath,
                                   os.path.basename(numrel_data).replace('.h5', '')
                                   + '_iota-%f' % iota + '_phi-%f' % phi + '_GEOM.png')
        plt.savefig(figname)

    if show_plot:
        plt.show()

    return fig


# Helper function that checks whether a number is essentially an integer.
# This number can be float or int.
def is_int(x):
    x = float(x)
    return x == float(np.floor(x))


# Helper function that checks if step - start == an integer * step.
# It also does basic sanity checks.
def check_step(start, end, step, var_name, bound=None):
    if start < 0 or end < 0 or step < 0:
        raise Exception("Not all input angles of {} are non-negative.".format(var_name))
    if start > end:
        raise Exception("End value of {} is smaller than start value.".format(var_name))
    if bound != None:
        if end > bound:
            raise Exception("End value of {} cannot be larger than {}.".format(var_name, bound))

    if float(step) == 0.0:
    # Avoiding zero division error.
        if not float(end - start) == 0.0:
            raise Exception("Step of {} is not consistent.".format(var_name))
    else:
        num_steps = float(end - start) / step
        if not is_int(num_steps):
            raise Exception("Step of {} is not consistent.".format(var_name))


# Helper function that generates the time-frequency maps (phyical and geometric units)
# for one numrel simulation at a pair of inclination angle and phase angle.
def gen_one_map(numrel_data, iota, phi, mother_freq, max_scale,
                name_in_degree, show_plot=False, save_plot=True,
                savepath=None):
    wf_data = gen_waveform(numrel_data, iota, phi)
    tf_data = tf_decompose(wf_data['hp'], wf_data["sample_times"], mother_freq, max_scale)


    return plot_physical(numrel_data, iota, phi,
                  wf_data["sample_times"], wf_data["hp"], wf_data["amp"],
                  wf_data["foft"], tf_data["wfreqs"], tf_data["wplane"],
                  max_scale,
                  show_plot=show_plot, in_degree=name_in_degree,
                  save_plot=save_plot, savepath=savepath)
#     plot_geometric(waveformName, iota, phi,
#                   wf_data["sample_times"], wf_data["hp"], wf_data["amp"],
#                   wf_data["foft"], tf_data["wfreqs"], tf_data["wplane"],
#                   maxScale,
#                   show_plot=show_plot, in_degree=name_in_degree,
#                   save_plot=save_plot, maxScale=maxScale, savepath=savepath)


# This function generates and saves a set of time-frequency maps for a set of
# inclination and phase angles.
def gen_maps(numrel_data_list, iota_start, iota_end, iota_step,
             phi_start, phi_end, phi_step, name_in_degree,
             save_plot=True, show_plot=False, savepath=None):
    # Checking whether the steps of iota and phi are consistent with the respective
	# start and end values.
    check_step(iota_start, iota_end, iota_step, "iota")
    check_step(phi_start, phi_end, phi_step, "phi")

    for numrel_data in numrel_data_list:
        iota = iota_start
        while iota <= iota_end:
            phi = phi_start
            while phi <= phi_end:
#                 print "current waveformName: ", waveformName, "iota: ", iota, "phi:", phi
                gen_one_map(numrel_data, iota, phi, mother_freq, max_scale,
                            name_in_degree, save_plot=save_plot, show_plot=show_plot,
                           savepath=savepath)
#                 print ""
                phi += phi_step

                # In case of phi_step == 0
                if phi == phi_start:
                    break
            iota += iota_step

            # In case of iota_step == 0
            if iota == iota_start:
                break


# Produce contour maps for the time-freq representations.
# Example:
# waveformName = "/waves/GT0577.h5"
# wf_data = gen_waveform(waveformName, iota, phi)
# tf_data = tf_decompose(wf_data['hp'], wf_data["sample_times"], motherFreq, maxScale)
# plot_contour(waveformName, iota, phi,
#              wf_data["sample_times"], tf_data["wfreqs"], tf_data["wplane"],
#             maxScale, save_plot=True, show_plot=False, savepath=savepath)
def plot_contour(numrel_data, iota, phi,
                sample_times, wfreqs, wplane, max_scale, scale="physical",
                show_plot=True, in_deg=True, save_plot=False,
                savepath=None):
    print "In plot_contour: {}, iota: {:.2f}, phi: {:.2f}".format(scale, iota, phi)
    NUM_LEVELS = 5
    fig, ax = plt.subplots(figsize=(fig_width, fig_height))
    cs = ax.contour(sample_times, wfreqs, abs(wplane), NUM_LEVELS)

    ax.set_xlim(-0.05,0.05)
    ax.set_ylim(0,max_scale)
    ax.set_xlabel(r'$\textrm{Time (s)}$')
    ax.set_ylabel(r'$\textrm{Frequency (Hz)}$')
    cb = plt.colorbar(cs, spacing="proportional")

    # The 0th child of the axes of color bar is a LineCollection object
    # which represents the color ticks on the color bar.
    cb.ax.get_children()[0].set_linewidths(10)

    if save_plot:
        if savepath == None:
            raise Exception("Savepath not supplied.")
        head = os.path.join(savepath, os.path.basename(numrel_data).replace(".h5", ""))
        iota_str = utils.ang_to_str(iota, to_deg=in_deg)
        phi_str = utils.ang_to_str(phi, to_deg=in_deg)
        figname = "{}_iota-{}_phi-{}.png".format(head, iota_str, phi_str)
#         print figname
        fig.savefig(figname)

    if show_plot:
        plt.show()

    return fig