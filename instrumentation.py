# instrumentation.py
# All the instrument/network classes

import os
import numpy as np
import pymap3d as pm
import pydarn

class ASINetwork(object):
    def __init__(self, network, label=None, elev=None, color=None, alt=None):
        filename = os.path.join(os.path.dirname(__file__), 'site_data', '{}_sites.txt'.format(network))
        sites = np.loadtxt(filename)
        self.label = label
        self.color = color

        self.sites = [ASI(site[0], site[1], site[2], elev, color, alt) for site in sites]


class ASI(object):
    def __init__(self, site_lat, site_lon, site_alt, elev, color, alt):
        self.site_lat = site_lat
        self.site_lon = site_lon
        self.site_alt = site_alt
        self.color = color
        # self.elev = elev

        self.lat, self.lon = self.generate_fov(elev, alt)

    def generate_fov(self, elev, alt):
        az = np.linspace(0., 360., 50)*np.pi/180.
        el = np.full(50, elev)*np.pi/180.
        lat, lon, alt = projected_beam(self.site_lat, self.site_lon, self.site_alt, az, el, proj_alt=alt)
        return lat[::-1], lon[::-1]


class AMISR(object):
    # def __init__(self, site_lat, site_lon, site_alt, site_name, color):
    def __init__(self, radar=None, label=None, color=None):

        site_coords = {'PFISR':[65.12992, -147.47104, 0.213], 'RISR-N':[74.72955, -94.90576, 0.145], 'RISR-C':[74.72955, -94.90576, 0.145]}
        self.site_lat, self.site_lon, self.site_alt = site_coords[radar]
        self.color = color
        self.label = label

        az, el = self.load_data(radar)
        self.lat, self.lon = self.generate_fov(az, el)

    def load_data(self, radar):
        filename = os.path.join(os.path.dirname(__file__), 'site_data', '{}GratingLimits.txt'.format(radar.replace('-','').lower()))
        data = np.loadtxt(filename)
        az = data[:,0]*np.pi/180.
        el = data[:,1]*np.pi/180.
        return az, el

    def generate_fov(self, az, el):
        lat, lon, alt = projected_beam(self.site_lat, self.site_lon, self.site_alt, az, el, proj_alt=450)
        return lat[::-1], lon[::-1]

class ISR(object):
    def __init__(self, site_lat, site_lon, site_alt, label=None, elev=None, color=None):
        self.site_lat = site_lat
        self.site_lon = site_lon
        self.site_alt = site_alt
        self.label = label
        self.color = color

        self.lat, self.lon = self.generate_fov(elev)

    def generate_fov(self, elev):
        az = np.linspace(0., 360., 50)*np.pi/180.
        el = np.full(50, elev)*np.pi/180.
        lat, lon, alt = projected_beam(self.site_lat, self.site_lon, self.site_alt, az, el, proj_alt=450)
        return lat[::-1], lon[::-1]

class SuperMAG(object):
    def __init__(self, color=None, label=None):
        self.color = color
        self.label = label
        filename = os.path.join(os.path.dirname(__file__), 'site_data', 'SuperMAG_sites.txt')
        data = np.loadtxt(filename,skiprows=44,usecols=(1,2))
        self.sites = np.array([data[:,1],data[:,0],np.zeros(data.shape[0])]).T

class SDNetwork(object):
    def __init__(self, color=None, label=None):
        self.color = color
        self.label = label

        radar_info = pydarn.SuperDARNRadars.radars
        self.sites = [SuperDARN(stid) for stid in radar_info.keys()]


class SuperDARN(object):
    def __init__(self, stid):

        gate_lat, gate_lon = pydarn.Coords.GEOGRAPHIC(stid)
        self.lat = np.concatenate((gate_lat[0,:],gate_lat[:,-1],gate_lat[-1,::-1],gate_lat[::-1,0]))
        self.lon = np.concatenate((gate_lon[0,:],gate_lon[:,-1],gate_lon[-1,::-1],gate_lon[::-1,0]))



def projected_beam(lat0,lon0,alt0,az,el,proj_alt=300.):

#     lat0, lon0, alt0 = site
#     az = az*np.pi/180.
#     el = el*np.pi/180.

    x, y, z = pm.geodetic2ecef(lat0, lon0, alt0*1000.)
    vx, vy, vz = pm.enu2uvw(np.cos(el)*np.sin(az), np.cos(el)*np.cos(az), np.sin(el), lat0, lon0)

    earth = pm.Ellipsoid()
    a2 = (earth.semimajor_axis + proj_alt*1000.)**2
    b2 = (earth.semimajor_axis + proj_alt*1000.)**2
    c2 = (earth.semiminor_axis + proj_alt*1000.)**2

    A = vx**2/a2 + vy**2/b2 + vz**2/c2
    B = x*vx/a2 + y*vy/b2 + z*vz/c2
    C = x**2/a2 + y**2/b2 + z**2/c2 -1

    alpha = (np.sqrt(B**2-A*C)-B)/A

    lat, lon, alt = pm.ecef2geodetic(x + alpha*vx, y + alpha*vy, z + alpha*vz)

    return lat, lon, alt/1000.
