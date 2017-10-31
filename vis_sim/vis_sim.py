"""
vis_sim.py
=============

Routines for visibility simulation using
a sky model and an antenna beam model
"""

# Load Modules
import matplotlib
import matplotlib.pyplot as plt
from mpl_toolkits.basemap import Basemap
import numpy as np
import healpy as hp
import os
import sys
from scipy import interpolate
import astropy.io.fits as fits
from astropy.time import Time
import ephem
from hera_cal import omni
import aipy
from collections import OrderedDict
from astropy.stats import biweight_midvariance
import copy
from pyuvdata import UVData, UVBeam
import itertools
from .helper import Helper
from .sky_model import Sky_Model
from .beam_model import Beam_Model



class Vis_Sim(Sky_Model, Beam_Model):
    """
    Vis_Sim
    -------

    class for handling visibility simulations

    """

    def init_sky(self, skyfile, **kwargs):
    	"""
    	initialize sky model. See Sky_Model class doc-string for info on arguments
    	"""
    	super(Vis_Sim, self).__init__(skyfile, **kwargs)

    def init_beam(self, beamfile, **kwargs):
        """
        initialize beam. See Beam_Model class doc-string for info on arguments
        """
        # initialize Beam_Model class
        super(Sky_Model, self).__init__(beamfile, **kwargs)

        # get polarization parameters
        self.pols = self.beam_pols.copy()
        self.Npols = len(self.pols)

        # set fiducial beam models
        self.fid_beam_models = copy.deepcopy(self.beam_models)

        # Initialize principal components
        self.Npcomps = 1
        self.pcomps  = np.ones((self.Npols, self.Nfreqs, self.Npcomps, self.beam_npix))

        # Initialize beam coefficients for each antenna
        self.beam_coeffs = np.zeros((self.red_info.nAntenna, self.Npols, self.Nfreqs, self.Npcomps))


    def __init__(self, calfile, skyfile, beamfile, info_kwargs={}, sky_nside=None, freqs=None,
                 onepol=False, pol=0, verbose=False):
        """
        Visibility Simulation

        Input:
        ------
        calfile : str
                name of an aipy calfile w/o .py suffix
                within your current path

        skyfile : str
            data for skyfile in .npz format

        beamfile : str
            data for beamfile as a pyuvdata .beamfits file

        info_kwargs : dictionary, default={}
            kwargs to feed omni.aa_to_info() function

        sky_nside : int, default=None
            nside healpix resolution parameter for sky model

        freqs : ndarray, default=None
            desired visibility frequency channels in MHz
            by default uses sky model frequency channels

        onepol : bool, default=False
            if True: solve only one auto-pol (faster)
                takes trace/2 of sky models (stokes I) and [pol] of beam models
            if False: solve full jones matrix for 4-pol terms (slower)

        pol : int, default=0
            if onepol is True, which polarization index to use for beam models (0 or 1)

        verbose : bool, default=False
            print out progress

        """
        self.print_message("...initializing visibility simulation", type=1, verbose=verbose)

        # Initialize sky
        self.init_sky(skyfile, sky_nside=sky_nside, freqs=freqs, verbose=verbose, onepol=onepol)

        # Assign variables
        self.calfile     = calfile
        self.info_kwargs = info_kwargs
        if freqs is None:
            self.freqs   = self.sky_freqs
        else:
            self.freqs   = freqs
        self.Nfreqs      = len(self.freqs)

        # Get Redundant Info
        self.print_message("...getting redundant info", type=1, verbose=verbose)
        self.AntArr   = aipy.cal.get_aa(calfile, self.freqs)
        self.red_info = omni.aa_to_info(self.AntArr, **info_kwargs)
        self.red_bls  = np.array(self.red_info.get_reds()) 
        self.ant_nums = np.unique(np.concatenate(self.red_bls.tolist()).ravel())
        self.Nants    = len(self.ant_nums)
        self.ant_pos  = OrderedDict([(str(i), self.red_info.antloc[self.red_info.ant_index(i)]) for i in self.ant_nums])
        self.ant2ind  = OrderedDict(zip(np.array(self.ant_nums,str), np.arange(self.Nants)))
 
        # Assign location info
        self.loc           = ephem.Observer()
        self.loc.lon       = self.AntArr.lon
        self.loc.lat       = self.AntArr.lat
        self.loc.elev      = self.AntArr.elev

        # Initialize fiducial beam models if beamfile
        self.print_message("...initializing beam models", type=1, verbose=verbose)
        self.init_beam(beamfile, loc=self.loc, pol=pol, onepol=onepol, freqs=self.freqs,
                       verbose=verbose, beam_nside=sky_nside)

    def build_beams(self, ant_inds=None, output=False, one_beam_model=True):
    	""""
    	build each antenna's beam model from fiducial model and principal components

    	Input:
    	------
    	output : bool, default=False
    		if True, return results

    	Result:
    	-------
    	self.ant_beam_models : dictionary
    		contains each antenna name in str as key
    		holding healpix beam model
    	"""
        # get antenna indices
        if ant_inds is None:
            ant_inds = np.arange(self.Nants)            

        # get each antennas beam model
        if one_beam_model == True:
            self.ant_beam_models = self.fid_beam_models.copy()
        else:
            self.ant_beam_models = np.array(map(lambda i: np.einsum('ijkl,ijk->ijl', self.pcomps, self.beam_coeffs[i]) + self.fid_beam_models, ant_inds))

    	if output == True:
    		return self.ant_beam_models


    def generate_vis_noise(self, Tnoise, freqs, beam_sa, size, bandwidth=1e6, int_time=10.7):
        """
        Generate vis noise via radiometer equation
        """
        Vnoise_jy = Tnoise * self.T2jy(freqs, beam_sa) / (bandwidth * int_time)
        return np.random.normal(loc=0.0, scale=1/np.sqrt(2), size=size) * Vnoise_jy


    def sim_obs(self, bl_array, JD_array, pool=None,
                write_miriad=False, fname=None, clobber=False, one_beam_model=True,
                interp=True, Tnoise=None, fast_noise=False):
        """
		Simulate a visibility observation of the sky

		Input:
		------
		bl_array : list, shape=(Nbls, 2), entry_format=tuple(int, int)
		    list of baselines (antenna pairs)
		    in (ant1, ant2) with type(ant1) == int

		JD_array : list, shape=(Ntimes,), dtype=float
			list of Julian Dates for observations
    	"""
        # get array info
        Nbls         = len(bl_array)
        str_bls      = [(str(x[0]), str(x[1])) for x in bl_array]
        Ntimes       = len(JD_array)
        rel_ants     = np.unique(np.concatenate(bl_array))         # relevant antennas
        rel_ant2ind  = OrderedDict(zip(np.array(rel_ants,str), np.arange(len(rel_ants))))

        # get antenna indices
        ant_inds = map(lambda x: self.ant2ind[str(x)], rel_ants)

        # get antenna positions for each baseline
        ant1_pos = np.array(map(lambda x: self.ant_pos[x[0]], str_bls))
        ant2_pos = np.array(map(lambda x: self.ant_pos[x[1]], str_bls))

        # get vector in u-v plane
        wavelengths = 2.9979e8 / (self.freqs * 1e6)
        self.uv_vecs = (ant2_pos - ant1_pos).reshape(Nbls, 3, -1) / wavelengths

        # Get direction unit vector, s-hat
        x, y, z = hp.pix2vec(self.beam_nside, np.arange(self.beam_npix))
        self.s_hat = np.array([y, x, z])

        # get phase map in topocentric frame (sqrt of full phase term)
        self.phase = np.exp( -1j * np.pi * np.einsum('ijk,jl->ikl', self.uv_vecs, self.s_hat) )

        # build beams for each antenna in topocentric coordinates
        self.build_beams(ant_inds=ant_inds, one_beam_model=one_beam_model)

        # create phase and beam product maps (Npol, Nbl, Nfreqs, Npix)
        self.beam_phs = self.ant_beam_models.reshape(self.Npols, -1, self.Nfreqs, self.beam_npix) * self.phase

        # loop over JDs
        self.vis_data = []
        for jd in JD_array:

            # map through each antenna and project polarized, multi-frequency beam model onto sky
            if one_beam_model == True:
                self.proj_beam_phs = self.project_beams(jd, self.sky_theta, self.sky_phi, beam_models=self.beam_phs, output=True, interp=interp)

            else:
                #proj_ant_beam_models = self.project_beams(JD, self.sky_theta, self.sky_phi,
                #beam_models=np.array(map(lambda x: self.ant_beam_models[str(x)], self.ant_nums)), output=True)
                self.proj_beam_phs = self.project_beams(jd, self.sky_theta, self.sky_phi, beam_models=self.beam_phs, output=True, interp=interp)

            # calculate visibility
            if pool is None:
                M = map
            else:
                M = pool.map

            # add noise
            if Tnoise is None or fast_noise is True:
                sky_models = self.sky_models
            else:
                sky_models = self.sky_models + self.generate_noise_map(Tnoise, self.sky_models.shape)

            if one_beam_model is True:
                if self.onepol is True:
                    vis = np.einsum("ijkl, ikl -> ijk", self.proj_beam_phs**2, sky_models)
                else:
                    vis = np.einsum("ijkl, imkl, mjkl -> imjk", self.proj_beam_phs, sky_models, self.proj_beam_phs)
            else:
                vis = np.array(map(lambda x: np.einsum("ijk, jk, ljk -> ilj", self.proj_beam_phs[rel_ant2ind[x[1][0]]],
                      sky_models, self.proj_beam_phs[rel_ant2ind[x[1][1]]]), enumerate(str_bls)))

            self.vis_data.append(vis)

        if self.onepol is True:
            self.vis_data = np.moveaxis(np.array(self.vis_data), 0, 2)
        else:
            self.vis_data = np.moveaxis(np.array(self.vis_data), 0, 3)
        
        # normalize by the effective-beam solid angle
        if one_beam_model == True:
            self.ant_beam_sa = np.repeat(np.einsum('ijk,ijk->ij', self.ant_beam_models, self.ant_beam_models)[ :, np.newaxis, :], Nbls, 1)
        else:
            self.ant_beam_sa = np.array(map(lambda x: np.einsum('ijk,ljk->ilj', self.proj_ant_beam_models[rel_ant2ind[x[1][0]]],
                                                            self.proj_ant_beam_models[rel_ant2ind[x[1][1]]]), enumerate(str_bls)))
        if self.onepol is True:
            self.vis_data /= self.ant_beam_sa[:, :, np.newaxis, :]
            self.vis_data_shape = "(1, Nbls, Ntimes, Nfreqs)"
        else:
            self.vis_data /= self.ant_beam_sa[:, :, :, np.newaxis, :]
            self.vis_data_shape = "(2, 2, Nbls, Ntimes, Nfreqs)"

        if fast_noise is True and Tnoise is not None:
            self.vis_data += self.generate_vis_noise(Tnoise, self.freqs*1e6, self.ant_beam_sa.squeeze(), self.vis_data.shape,
                                                     bandwidth=self.bandwidth*1e6, int_time=10.7)

        # write to file
        if write_miriad == True:

            # create uvdata object
            uvd = UVData()

            uvd.Nants_telescope    = self.Nants
            uvd.Nants_data         = len(np.unique(bl_array))
            uvd.Nbls               = Nbls
            uvd.Ntimes             = Ntimes
            uvd.Nblts              = Nbls * Ntimes
            uvd.Nfreqs             = self.Nfreqs
            uvd.Npols              = self.Nxpols
            uvd.Nspws              = 1
            uvd.antenna_numbers    = self.ant_nums
            uvd.antenna_names      = np.array(self.ant_nums, dtype=str)
            uvd.ant_1_array        = np.concatenate([map(lambda x: x[0], bl_array) for i in range(Ntimes)])
            uvd.ant_2_array        = np.concatenate([map(lambda x: x[1], bl_array) for i in range(Ntimes)])
            uvd.baseline_array     = np.concatenate([np.arange(Nbls) for i in range(Ntimes)])
            uvd.freq_array         = (self.freqs * 1e6).reshape(1, -1)
            uvd.time_array         = np.repeat(np.array(JD_array)[:, np.newaxis], Nbls, axis=1).ravel()
            uvd.channel_width      = (self.freqs * 1e6)[1] - (self.freqs * 1e6)[0]
            uvd.data_array         = self.vis_data.reshape(uvd.Nblts, 1, self.Nfreqs, self.Nxpols)
            uvd.flag_array         = np.ones_like(uvd.data_array, dtype=np.bool)
            uvd.history            = " "
            uvd.instrument         = " "
            uvd.integration_time   = 10.7
            uvd.lst_array          = np.repeat(np.array(map(lambda x: self.JD2LST(self.loc, x), JD_array))[:, np.newaxis], Nbls, axis=1).ravel()
            uvd.nsample_array      = np.ones_like(uvd.data_array, dtype=np.float)
            uvd.object_name        = ""
            uvd.phase_type         = "drift"
            uvd.polarization_array = np.array(map(lambda x: {"XX":-5,"YY":-6,"XY":-7,"YX":-8}[x], self.xpols))
            uvd.spw_array          = np.array([0])
            uvd.telescope_location = np.array([ 6378137.*np.cos(self.loc.lon)*np.cos(self.loc.lat),
                                                6378137.*np.sin(self.loc.lon)*np.cos(self.loc.lat),
                                                6378137.*np.sin(self.loc.lat) ])
            uvd.telescope_name     = " "
            uvd.uvw_array          = np.ones((uvd.Nblts, 3))
            uvd.vis_units          = "Jy"
            zen_dec, zen_ra        = self.loc.radec_of(0, np.pi/2)
            uvd.zenith_dec         = np.ones(uvd.Nblts) * zen_dec
            uvd.zenith_ra          = np.ones(uvd.Nblts) * zen_ra

            uvd.write_miriad(fname, clobber=clobber)


































