""""
sky_model.py
------------

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


class Sky_Model(Helper):
    """
    Sky_Model
    --------------

    class for handling of sky model data
    """
    def __init__(self, skyfile, sky_nside=None, freqs=None, verbose=False, onepol=False):
        """
        load sky model
        and optionally interpolate to specific nside
        spatial resolution and frequency resolution

        Input:
        -------
        skyfile : str
            path to sky data file as a .npz file
            with "sky" as multifrequency sky and
            "freqs" as frequency channels in MHz.
            sky is assumed to be in galactic coordinates.

        sky_nside : int, default=None
            downsample sky to desired healpix nside
            resolution (not recommended).
            If None, default is what the skyfile provides

        freqs : ndarray, default=None
            frequency channels of sky model in MHz. If None, default
            is what the skyfile provides

        onepol : bool, default=False
            if True, keep only stokes I map
            if False, keep full Jones matrix

        Result:
        -------
        self.sky_models : ndarray, shape=(2, 2, Nfreqs, Npix)
            first two axes are for 4-pol stokes parameters
            [ [I+Q, U+iV], [U-iV, I-Q] ]
            (multi-frequency) sky sky model in healpix
            using RING ordering in units of MJy-str

        self.freqs : ndarray, shape=(Nfreqs,)
            frequency of sky maps in MHz

        self.theta : ndarray, shape=(Npix,)
            angle of latitude of healpix map in radians
            in galactic coordinates

        self.phi : ndarray, shape=(Npix,)
            angle of longitude of healpix map in radians
            in galactic coordinates
        """

        # Load Sky 
        sky_data    = np.load(skyfile)
        sky_models  = sky_data['sky']
        sky_freqs   = sky_data['freqs']
        sky_freq_ax = 2

        # check sky_models are appropriate shape
        if len(sky_models.shape) != 4:
            raise ValueError("sky_models.shape = {} when it should have len = 4".format(sky_models.shape))

        # check for onepol
        if onepol is True:
            sky_models = (np.trace(sky_models, axis1=0, axis2=1) / 2.0)[np.newaxis]
            sky_freq_ax -= 1

        # 1D interpolation of frequency axis
        if freqs is not None:
            sky_models = interpolate.interp1d(sky_freqs, sky_models, axis=sky_freq_ax)(freqs)
            sky_freqs = freqs

        # get theta and phi arrays for default data
        default_sky_nside = hp.npix2nside(sky_models.shape[sky_freq_ax+1])
        theta, phi = hp.pix2ang(default_sky_nside, np.arange(12*default_sky_nside**2), lonlat=False)

        # spatially interpolate healpix if desired
        if sky_nside is not None:
            # down sample
            theta, phi = hp.pix2ang(sky_nside, np.arange(12*sky_nside**2))
            sky_models = self.healpix_interp(sky_models, default_sky_nside, theta, phi)
            default_sky_nside = sky_nside

        # Assign variables to class
        self.sky_nside  = default_sky_nside
        self.sky_npix   = 12 * default_sky_nside**2
        self.sky_models = sky_models
        self.sky_freqs  = sky_freqs
        self.sky_theta  = theta
        self.sky_phi    = phi
        self.onepol     = onepol
        self.sky_freq_ax = sky_freq_ax

    def sky_freq_interp(self, freqs):
        """
        1D interpolation of sky models along frqeuency axis

        freqs : ndarray
            frequency channels in MHz
        """
        self.sky_models = interpolate.interp1d(self.sky_freqs, self.sky_models, axis=self.sky_freq_ax)(freqs)
        self.sky_freqs = freqs

    def plot_sky(self, loc, skymap, ax=None, log10=True, res=300, axoff=True, cbar=True,
                    save=False, fname=None, plot_kwargs={'cmap':'viridis'},
                    cbar_kwargs={}, basemap=True, rot=None):
        """
        Plot Sky Model in orthographic coordinates given observer
        location and date

        Input:
        ------
        loc : ephem location object
            relevant subattributes are loc.lon, loc.lat and loc.date

        skymap : ndarray, shape=(Npix,)
            sky map in healpix RING ordered

        ax : matplotlib axis object, default=None
            feed a previously defined axis if desired
            else create a new figure and axis object

        log10 : bool, default=True
            take the log10 of the map before plotting

        res : int, default=300
            polar coordinates pixel resolution
            plotting gets slow when res > 500

        axoff : bool, default=True
            turn off axes tick labels

        cbar : bool, default=True
            make a colorbar

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
        # rotate map
        if rot is not None:
            rot = self.rotate_map(self.sky_nside, rot=rot)
            skymap = skymap[rot]

        # get ra dec of location
        obs_ra, obs_dec = loc.radec_of(0, np.pi/2.0)

        # get rotation sorting array
        rot = self.rotate_map(self.sky_nside, rot=[obs_ra, obs_dec-np.pi/2], coord=['G', 'C'], theta=self.sky_theta, phi=self.sky_phi)

        # get rotated sky
        rot_sky = skymap[rot]

        # Get polar theta and r
        omega, r = np.meshgrid(np.linspace(0,2*np.pi,res), np.linspace(0,np.pi/2.0,res))

        # sample sky at polar coordinates
        obs_sky_polar = hp.get_interp_val(rot_sky, r.ravel(), omega.ravel()).reshape(res,res)

        # rotate omega
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
            obs_sky_polar = np.log10(obs_sky_polar)

        # use basemap
        if basemap == True:
            bmap = Basemap(projection='ortho',lat_0=90,lon_0=-90, ax=ax)

            # turn r from co-latitude to latitude
            r = np.abs(r-np.pi/2)

            # get x, y arrays for basemap
            x, y = bmap(omega*180/np.pi, r*180/np.pi)

            # plot
            cax = bmap.pcolor(x, y, obs_sky_polar, **plot_kwargs)

        # use polar
        else:
            # plot
            cax = ax.pcolormesh(omega, r, obs_sky_polar, **plot_kwargs)

        # turn axis off
        if axoff == True:
            ax.get_xaxis().set_visible(False)
            ax.get_yaxis().set_visible(False)

        # add colorbar
        if cbar == True:
            plt.colorbar(cax, orientation='vertical', **cbar_kwargs)

        if save == True:
            plt.savefig(fname, dpi=200, bbox_inches='tight')

        if custom_fig == True:
            return fig

