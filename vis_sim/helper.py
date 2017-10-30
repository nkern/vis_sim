""""
helper.py
----------

Helper class and functions
"""
import healpy as hp
import numpy as np


class Helper(object):
    """
    Helper
    -------

    Helper routines
    """
    def print_message(self, message, type=0, verbose=True):
        """
        print message to stdout
        """
        if verbose == True:
            if type == 0:
                print("\n%s" % message)
            elif type == 1:
                print("\n%s\n%s" % (message, '-'*40))

    def JD2LST(self, JD, longitude=None, loc=None):
        """
        use astropy to convert from JD to LST
        provide either:
        loc : ephem Observer instance
        longitude : float, longitude in degrees East
        """
        if loc is not None:
            t = Time(JD, format="jd")
            lst = t.sidereal_time('apparent', longitude=loc.lon*180/np.pi)
        elif longitude is not None:
            t = Time(JD, format="jd")
            lst = t.sidereal_time('apparent', longitude=longitude)
        return lst.value

    def healpix_interp(self, maps, map_nside, theta, phi, nest=False):
        """
        healpix map bi-linear interpolation

        Input:
        maps : ndarray, shape=(Nmaps, Npix)
            array containing healpix map(s)

        map_nside : int
            nside parameter for maps

        theta : ndarray, shape=(Npix)
            1D array containing co-latitude of
            desired inteprolation points in radians

        phi : ndarray, shape=(Npix)
            1D array containing longitude of
            desired interpolatin points in radians

        """
        if nest == True:
            r = hp._healpy_pixel_lib._get_interpol_nest(map_nside, theta, phi)
        else:
            r = hp._healpy_pixel_lib._get_interpol_ring(map_nside, theta, phi)
        p=np.array(r[0:4])
        w=np.array(r[4:8])
        if maps.ndim == 2:
            return np.einsum("ijk,jk->ik", maps[:, p], w)
        elif maps.ndim == 3:
            return np.einsum("hijk,jk->hik", maps[:, :, p], w)
        elif maps.ndim == 4:
            return np.einsum("ghijk,jk->ghik", maps[:, :, :, p], w)

    def rotate_map(self, nside, rot=None, coord=None, theta=None, phi=None, interp=False,
                   inv=True):
        """
        rotate healpix map between coordinates and/or in longitude and latitude

        nside : int, nside resolution of map

        rot : list, length=2
            rot[0] = longitude rotation (radians)
            rot[1] = latitude rotation (radians)

        coord : list, length=2
            transformation between coordinate systems
            see healpy.Rotator for convention

        theta : co-latitude map in alt-az coordinates

        phi : longitude map in alt-az coordinates

        interp : bool, default=False
            if True, use interpolation method (healpy.get_interp_val)
            if False, use slicing method (healpy.ang2pix)

        inv : bool, default=True
            keyword to feed hp.Rotator object
            to go from galactic to topocentric, inv=True
            to go from topocentric to galactic, inv=False
        """
        # if theta and phi arrays are not fed, build them
        if theta is None or phi is None:
            theta, phi = hp.pix2ang(nside, np.arange(hp.nside2npix(nside)))

        # get rotation
        rot_theta, rot_phi = hp.Rotator(rot=rot, coord=coord, inv=inv, deg=False)(theta, phi)

        if interp is False:
            # generate pixel indices array
            pix = np.arange(hp.nside2npix(nside))[hp.ang2pix(nside, rot_theta, rot_phi)]
            return pix

        else:
            return rot_theta, rot_phi
