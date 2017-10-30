"""
beam_model.py
-------------


"""
import matplotlib
import matplotlib.pyplot as plt
from mpl_toolkits.basemap import Basemap
import numpy as np
import healpy as hp
from collections import OrderedDict
import os
import sys
from scipy import interpolate
import astropy.io.fits as fits
from astropy.time import Time
import ephem
from pyuvdata import UVData, UVBeam
from .helper import Helper


class Beam_Model(Helper):
    """
    Beam_Model
    ----------

    class for handling of beam models
    """
    def __init__(self, beamfile, loc=None, freqs=None, pool=None, verbose=False,
                 mask_hor=True, beam_nside=None, onepol=False, pol=0, beam_type='intensity_sq'):
        """
        Load and configure beam models

        Input:
        ------
        beamfile : str
            pyuvdata beamfits file, holding a healpix beam in topocentric coordinates

        freqs : ndarray, default=None
            frequency channels in MHz, defaults to
            channels in steps of 1 MHz

        beam_data_path : str
            path to beam data in fits format with 2 HDUs
            with healpix beam models in 0th hdu w/ shape=(Npix, Nfreq)
            and beam frequencies in 1st hdu w/ shape=(Nfreq,)

        beam_nside : int
            interpolate beam onto beam_nside

        onepol : bool, default=False
            if True: use only one feed polarization (zeroth axis)

        pol : int, default=0, option = 0 or 1
            if onepol is True, which pol index to use in beam_models 

        beam_type : str, default='power', options=['intensity', 'intensity_sq']
            specifies the beam_type of the map. the measurement equation
            reads V = A1.T * I * A2, where A is the beam intensity and I is the sky
            brightness. If we assume A1 == A2 and A3 == A2**2, then this
            simplifies to V = A3 * I, where A1 and A2 have type of 'intensity'
            and A3 has type of 'intensity_sq', i.e. intensity_sq = intensity**2.
            this simulation needs a beam type of 'intensiy', so if fed
            'intensity_sq', it will take the square root of the maps.

        Result:
        -------
        self.beam_models : masked array, shape=(Nfreq, Npix)
            beam model map in healpix with sub-horizon masked out

        self.beam_freqs : ndarray, shape=(Nfreqs,)
            ndarray containing frequencies of beam in MHz

        self.beam_theta : ndarray, shape=(Npix,)
            ndarray containing co-latitude in radians of
            healpix map

        self.beam_phi : ndarray, shape=(Npix,)
            ndarray containing longitude in radians of
            healpix map
        """
        # Assign telescope coordinates and location
        if loc is None:
            loc = ephem.Observer()
        self.loc = loc

        # Load Models in healpix from fits file
        uvb         = UVBeam()
        uvb.read_beamfits(beamfile)
        beam_models = uvb.data_array[0,0]
        beam_freqs  = uvb.freq_array.squeeze() / 1e6
        pol_arr     = uvb.polarization_array

        # ensure beam type is intensity
        if beam_type == 'intensity_sq':
            beam_models = beam_models ** (1./2)

        # ensure beam_models.shape
        if len(beam_models.shape) < 2 or len(beam_models.shape) > 3:
            raise ValueError("beam_models.shape is {}, but must have len of either 2 or 3".format(beam_models.shape))

        # select onepol
        if onepol is True:
            beam_models = beam_models[pol][np.newaxis]
            pol_arr = np.array(pol_arr[pol])[np.newaxis]
        else:
            if beam_models.shape[0] != 2:
                raise ValueError("beam_models.shape[0] must be 2, but is {}".format(beam_models.shape[0]))

        # 1D interpolation of frequency axis if desired
        if freqs is not None:
            beam_models = interpolate.interp1d(beam_freqs, beam_models, axis=1,
                                               bounds_error=False, fill_value='extrapolate')(freqs)
            beam_freqs = freqs

        # Get theta and phi arrays
        default_beam_nside = uvb.nside
        default_beam_npix = uvb.Npixels
        beam_theta, beam_phi = hp.pix2ang(default_beam_nside, np.arange(default_beam_npix), lonlat=False)

        # spatially interpolate healpix if desired
        if beam_nside is not None:
            # down sample
            default_beam_nside = beam_nside
            default_beam_npix = hp.nside2npix(default_beam_nside)
            beam_theta, beam_phi = hp.pix2ang(default_beam_nside, np.arange(default_beam_npix))
            beam_models = np.array(map(lambda x: map(lambda y: hp.get_interp_val(y, beam_theta, beam_phi), x), beam_models))

        # make sure boresight is normalized to 1
        beam_models /= np.max(beam_models, axis=-1)[:, :, np.newaxis]

        # mask beam models below horizon
        if mask_hor is True:
            mask = (beam_theta > np.pi/2)
            beam_models[:, :, mask] = 0.0

        # assign vars to class
        self.beam_nside  = default_beam_nside
        self.beam_npix   = default_beam_npix
        self.beam_models = beam_models
        self.beam_freqs  = beam_freqs
        self.beam_theta  = beam_theta
        self.beam_phi    = beam_phi
        self.beam_pols   = pol_arr

    def beam_freq_interp(self, freqs):
        """
        1D interpolation of beam models along frqeuency axis

        freqs : ndarray
            frequency channels in MHz
        """
        self.beam_models = interpolate.interp1d(self.beam_freqs, self.beam_models, axis=1)(freqs)
        self.beam_freqs = freqs

    def project_beams(self, JD, sky_theta, sky_phi, beam_models=None, obs_lat=None, obs_lon=None,
                        freqs=None, output=False, pool=None, interp=False):
        """
        Project beam models into healpix galactic coordinates
        given observer location, observation date and sky models
        and interpolate onto sky model healpix nside resolution

        Input:
        ------
        JD : float
            Julian date of observation in J2000 epoch

        sky_theta : ndarray, shape=(Npix,)
            co-latitude in radians of sky healpix map
            in galactic coordinates

        sky_phi : ndarray, shape=(Npix,)
            longitude in radians of sky healpix map
            in galactic coordinates

        beam_models : list, shape=(Npol, Nfreq, Npix)
            set of beam models, default=None
            in which case it will use self.beam_models

        obs_lat : str or float
            observer's latitude on Earth
            if str type: format "deg:hour:min"
            if float type: format is radians

        obs_lon : str or float
            observer's longitude on Earth
            if str type: format "deg:hour:min"
            if float type: format is radians

        Result:
        -------
        self.sky_beam_models : masked array, shape=(Nfreq, Npix)
            beam models projected onto galactic coordaintes
            and interpolated onto sky model healpix resolution
            at each frequency with sub-horizon masked out

        self.s_hat : ndarray, shape=(Npix,3)
            ndarray containing pointing unit vector in observer's
            cartesian frame for each healpix pixel of beam
        """
        # get beam models
        if beam_models is None:
            beam_models = self.beam_models 

        # Assign coordinates and date of observation
        self.loc.date = Time(JD, format='jd').datetime
        if obs_lon is not None:
            self.loc.lon = obs_lon
        if obs_lat is not None:
            self.loc.lat = obs_lat

        # get ra/dec of zenith
        obs_ra, obs_dec = self.loc.radec_of(0, np.pi/2.0)

        # get rotation sorting array
        if interp is True:
            rot_theta, rot_phi = self.rotate_map(self.sky_nside, rot=[obs_ra, obs_dec-np.pi/2], coord=['G', 'C'], theta=sky_theta, phi=sky_phi, inv=False, interp=interp)
            self.sky_beam_models = self.healpix_interp(beam_models, self.beam_nside, rot_theta, rot_phi)

        else:
            rot = self.rotate_map(self.sky_nside, rot=[obs_ra, obs_dec-np.pi/2], coord=['G', 'C'], theta=sky_theta, phi=sky_phi, inv=False, interp=interp)
            self.sky_beam_models = beam_models[:, :, rot]

        self.sky_beam_freqs = self.beam_freqs

        # Interpolate frequency axis if desired
        if freqs is not None:
            self.sky_beam_models = map(lambda sbm: interpolate.interp1d(self.beam_freqs, sbm, axis=0)(freqs), self.sky_beam_models)
            self.sky_beam_freqs = freqs

        if output == True:
            return self.sky_beam_models


    def plot_beam(self, beam, ax=None, log10=False, dBi=False, res=300, axoff=True, cbar=False,
                    contour=True, levels=None, nlevels=None, label_cont=False,
                    save=False, fname=None, basemap=True, verbose=False,
                    plot_kwargs={'cmap':'YlGnBu_r','linewidths':0.75,'alpha':0.8},
                    cbar_kwargs={}, rot=[0, np.pi/2]):
        """
        Plot beam model in orthographic coordinates

        Input:
        ------
        beam : ndarray, shape=(Npix,)
            beam response in healpix RING ordered

        ax : matplotlib axis object, default=None
            feed a previously defined axis if desired
            else create a new figure and axis object

        log10 : bool, default=False
            take the log10 of the map before plotting

        dBi : bool, default=False
            express data in deciBels of intensity
            in other words: log10(data) * 10

        res : int, default=300
            polar coordinates pixel resolution
            Beware: plotting gets slow when res > 500

        axoff : bool, default=True
            turn off axes tick labels

        cbar : bool, default=True
            make a colorbar

        contour : bool, default=True
            if True: contour plot
            else: pcolormesh plot

        levels : list, default=None
            levels for contours

        nlevels : int, default=None
            if levels is None, try to 
            make custom levels with nlevels
            if None and levels is None, contour()
            will make its own levels by default

        label_cont : bool, default=False
            label contours with level

        save : bool, default=False
            save image to file

        fname : str, default=None
            filename of image to save

        plot_kwargs : dictionary
            keyword arguments to feed plotting routine

        cbar_kwargs : dictionary
            kwargs to feed colorbar routine
        
        Output:
        -------
        if ax is None:
            outputs matplotlib.pyplot.figure object
        """
        if dBi == True and log10 == True:
            log10 = False
            if verbose == True:
                print "...both log10 dBi is True, using dBi"

        # rotate beam
        if rot is not None:
            rot = self.rotate_map(self.beam_nside, rot=rot)
            beam = beam[rot]

        # get beam_theta (co-latitude) and beam_phi (longitude)
        nside = int(np.sqrt(len(beam)/12.0))
        beam_theta, beam_phi = hp.pix2ang(nside, np.arange(len(beam)), lonlat=False)

        # rotate observed beam upwards
        theta_pol, phi_pol = hp.Rotator(rot=[0,np.pi/2,0],deg=False,inv=False)(beam_theta,beam_phi)
        rot_beam = hp.get_interp_val(beam, theta_pol, phi_pol)

        # Get polar theta and r
        omega, r = np.meshgrid(np.linspace(0,2*np.pi,res), np.linspace(0,np.pi/2,res))

        # sample sky at polar coordinates
        beam_polar = hp.get_interp_val(rot_beam, r.ravel(), omega.ravel()).reshape(res,res)

        # rotate omega by np.pi/2
        omega  = np.unwrap(omega + np.pi/2.0)

        # mirror about x-axis
        i = r * np.cos(omega) + 1j * r * np.sin(omega)
        i = np.conj(i)
        omega = np.unwrap(np.angle(i))
        r = np.abs(r)

        # create custom fig, ax if necessary
        custom_fig = False
        if ax is None:
            custom_fig = True
            if basemap == True:
                fig, ax = plt.subplots(1, figsize=(6,6))
            else:
                fig, ax = plt.subplots(1, figsize=(6,6), subplot_kw=dict(projection='polar'))

        # log data if desired
        if log10 == True:
            beam_polar = np.log10(beam_polar)

        if dBi == True:
            beam_polar = np.log10(beam_polar) * 10

        # use basemap
        if basemap == True:
            bmap = Basemap(projection='ortho',lat_0=90,lon_0=-90, ax=ax)

            # turn r from co-latitude to latitude
            r = np.abs(r-np.pi/2)

            # get x, y arrays for basemap
            x, y = bmap(omega*180/np.pi, r*180/np.pi)

            # plot
            if contour == True:
                # get contour levels
                if levels is None and nlevels is not None:
                    maxval = beam_polar.max() * 0.95
                    levels = [maxval/i for i in np.arange(1.0, nlevels+1)][::-1]
                cax = bmap.contour(x, y, beam_polar, levels=levels, **plot_kwargs)
            else:
                cax = bmap.pcolor(x, y, beam_polar, **plot_kwargs)
        # use polar
        else:
            # plot
            if contour == True:
                # get contour levels
                if levels is None and nlevels is not None:
                    maxval = beam_polar.max() * 0.95
                    levels = [maxval/i for i in np.arange(1.0, nlevels+1)][::-1]
                cax = ax.contour(omega, r, beam_polar, levels=levels, **plot_kwargs)
            else:
                cax = ax.pcolormesh(omega, r, beam_polar, **plot_kwargs)

        # turn axis off
        if axoff == True:
            ax.get_xaxis().set_visible(False)
            ax.get_yaxis().set_visible(False)

        # add colorbar
        if cbar == True:
            plt.colorbar(cax, orientation='horizontal', **cbar_kwargs)

        # label contours
        if label_cont == True:
            ax.clabel(cax, inline=1, fontsize=10)

        if save == True:
            plt.savefig(fname, dpi=200, bbox_inches='tight')

        if custom_fig == True:
            return fig

