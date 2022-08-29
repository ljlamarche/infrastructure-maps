# ground_facilities.py
# map relevant instumentation in northern and southern hemispheres

import numpy as np
import datetime as dt
import pymap3d as pm
import os
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
# from apexpy import Apex
import pydarn
import matplotlib.gridspec as gridspec
import cartopy.crs as ccrs
import cartopy.feature as cfeature


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



def map():

    fig = plt.figure(figsize=(10,10))
    gs = gridspec.GridSpec(1,1)
    gs.update(left=0.05,right=0.95,bottom=0.05,top=0.9,hspace=0.1)
    ax = plt.subplot(gs[0],projection=ccrs.Orthographic(central_longitude=-100,central_latitude=45))
    ax.set_extent([-150, -50, 15, 85], crs=ccrs.PlateCarree())
    # ax_NH.set_title('Northern Hemisphere')
    # ax_SH = plt.subplot(gs[1],projection=ccrs.SouthPolarStereo())
    # ax_SH.set_extent([-180,180,-65,-90],crs=ccrs.PlateCarree())
    # ax_SH.set_title('Southern Hemisphere')
    # for ax in [ax_NH,ax_SH]:
    ax.coastlines(resolution='50m',zorder=2)
    ax.gridlines()
    # ax.background_img()
    # ax.background_img(name='ETOPO', resolution='high')
    ax.background_img(name='BM',resolution='mid')

    # # plot magnetic grid
    # A = Apex(2014)
    # for mlat in [50,60,70,80,90]:
    #     grid_maglat = A.convert(mlat,np.linspace(0,360,100),'apex','geo',height=300)
    #     ax.plot(grid_maglat[1],grid_maglat[0],color='gold',linewidth=0.7,transform=ccrs.Geodetic())
    #     # grid_maglat = A.convert(-mlat,np.linspace(0,360,100),'apex','geo',height=300)
    #     # ax_SH.plot(grid_maglat[1],grid_maglat[0],color='gold',linewidth=0.7,transform=ccrs.Geodetic())
    # for mlon in [0,60,120,180,240,300]:
    #     grid_mlon = A.convert(np.linspace(30,90,100),mlon,'apex','geo',height=300)
    #     ax.plot(grid_mlon[1],grid_mlon[0],color='gold',linewidth=0.7,transform=ccrs.Geodetic())
    #     # grid_mlon = A.convert(-np.linspace(30,90,100),mlon,'apex','geo',height=300)
    #     # ax_SH.plot(grid_mlon[1],grid_mlon[0],color='gold',linewidth=0.7,transform=ccrs.Geodetic())

    # plot ASI
    mango = ASINetwork('MANGO', label='MANGO\nairglow\nimaging\nNetwork', elev=20., color='darkorange', alt=250)
    green = ASINetwork('GREEN', label='Boston University\nairglow imagers', elev=20., color='limegreen', alt=100)
    alaska = ASINetwork('ALASKA', label='Alaska\nauroral\nimagers', elev=20., color='darkblue', alt=250)
    themis = ASINetwork('THEMIS', label='THEMIS\nimaging\narray', elev=20., color='turquoise', alt=100)
    rego = ASINetwork('REGO', label='REGO\nairglow\nauroral\nimagers', elev=20., color='crimson', alt=250)
    for network in [mango, green, alaska, themis, rego]:
        for cam in network.sites:
            ax.plot(cam.lon, cam.lat, color=network.color, linewidth=1.5, zorder=6.5, transform=ccrs.Geodetic())
    ax.text(-132, 37, mango.label, color=mango.color, weight='heavy', zorder=7, horizontalalignment='center', verticalalignment='center', transform=ccrs.Geodetic())
    ax.text(-120, 30, green.label, color=green.color, weight='heavy', zorder=7, horizontalalignment='center', verticalalignment='center', transform=ccrs.Geodetic())
    ax.text(-162, 58, alaska.label, color=alaska.color, weight='heavy', zorder=7, horizontalalignment='center', verticalalignment='center', transform=ccrs.Geodetic())
    ax.text(-53, 48, themis.label, color=themis.color, weight='heavy', zorder=7, horizontalalignment='center', verticalalignment='center', transform=ccrs.Geodetic())
    ax.text(-70, 67, rego.label, color=rego.color, weight='heavy', zorder=7, horizontalalignment='center', verticalalignment='center', transform=ccrs.Geodetic())

    # plot SuperMAG sites
    sm = SuperMAG(color='violet', label='SuperMAG magnetometer Network')
    ax.scatter(sm.sites[:,1],sm.sites[:,0], color=sm.color, s=8, zorder=4, transform=ccrs.Geodetic())
    # ax_SH.scatter(sites[:,1],sites[:,0],color=mag_color,s=8,zorder=4,transform=ccrs.Geodetic())
    ax.text(-100, 18, sm.label, color=sm.color, weight='heavy', horizontalalignment='center', verticalalignment='center', transform=ccrs.Geodetic())


    # plot SuperDARN network:
    sd = SDNetwork(color='lightgrey', label='SuperDARN radar Network')
    for rad in sd.sites:
        ax.fill(rad.lon, rad.lat, color=sd.color, alpha=0.15, zorder=3, transform=ccrs.Geodetic())
        ax.plot(rad.lon, rad.lat, color=sd.color, linewidth=1, zorder=3, transform=ccrs.Geodetic())
    ax.text(-150, 52, sd.label, color=sd.color, weight='heavy', rotation=-47, horizontalalignment='center', verticalalignment='center', transform=ccrs.Geodetic())


    millstone = ISR(42.6233, -71.4882, 0.131, label='Millstone\nHill\nISR', elev=45, color='deeppink')
    aerecibo = ISR(18.34417, -66.75278, 0.497, label='Arecibo ISR', elev=71, color='darkorange')
    sondrestrom = ISR(66.9858, -50.945626, 0.196, label='Sondrestrom\nISR', elev=45, color='darkorchid')
    pfisr = AMISR('PFISR', label='Poker Flat\nISR', color='gold')
    risrn = AMISR('RISR-N', label='Res. Bay\nISR North', color='deepskyblue')
    risrc = AMISR('RISR-C', label='Res. Bay\nISR Canada', color='forestgreen')
    for radar in [millstone, pfisr, risrn, risrc]:
        ax.fill(radar.lon, radar.lat, color=radar.color, alpha=0.5, zorder=6, transform=ccrs.Geodetic())
        ax.plot(radar.lon, radar.lat, color=radar.color, zorder=6, transform=ccrs.Geodetic())
        ax.text(np.mean(radar.lon), np.mean(radar.lat), radar.label, color='k', weight='heavy', horizontalalignment='center', verticalalignment='center', zorder=6.5, transform=ccrs.Geodetic())


    plt.savefig('NSF_facilities.png')
#     plt.show()





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


def main():
    map()

if __name__ == '__main__':
    main()
