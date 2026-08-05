import numpy as np
import pymap3d as pm
import matplotlib.pyplot as plt
import os
import cartopy.crs as ccrs
import cartopy.feature as cfeature
import pydarn


def projected_beam(lat0, lon0, az, el, proj_alt=300.):

    x, y, z = pm.geodetic2ecef(lat0, lon0, 0.)
    vx, vy, vz = pm.enu2uvw(np.cos(el)*np.sin(az), np.cos(el)*np.cos(az), np.sin(el), lat0, lon0)

    #earth = pm.Ellipsoid()
    earth = pm.Ellipsoid.from_name('wgs84')
    a2 = (earth.semimajor_axis + proj_alt*1000.)**2
    b2 = (earth.semimajor_axis + proj_alt*1000.)**2
    c2 = (earth.semiminor_axis + proj_alt*1000.)**2

    A = vx**2/a2 + vy**2/b2 + vz**2/c2
    B = x*vx/a2 + y*vy/b2 + z*vz/c2
    C = x**2/a2 + y**2/b2 + z**2/c2 -1

    alpha = (np.sqrt(B**2-A*C)-B)/A

    lat, lon, alt = pm.ecef2geodetic(x + alpha*vx, y + alpha*vy, z + alpha*vz)

    return lat, lon, alt/1000.


def generate_asi_fov(site_lat, site_lon, elev, alt, npoint=50):
    az = np.linspace(0., 360., npoint)*np.pi/180.
    el = np.full(npoint, elev)*np.pi/180.
    lat, lon, alt = projected_beam(site_lat, site_lon, az, el, proj_alt=alt)
    return lat[::-1], lon[::-1]

def generate_fpi_beams(site_lat, site_lon, elev, alt):
    az = np.arange(0., 360., 90.)*np.pi/180.
    el = np.full(4, elev)*np.pi/180.
    lat, lon, alt = projected_beam(site_lat, site_lon, az, el, proj_alt=alt)
    return lat[::-1], lon[::-1]


def generate_amisr_fov(site_lat, site_lon, alt, radar):
    filename = os.path.join(os.path.dirname(__file__), 'site_data', '{}GratingLimits.txt'.format(radar.replace('-','').lower()))
    az, el = np.loadtxt(filename, usecols=[0,1], unpack=True)
    az = np.deg2rad(az)
    el = np.deg2rad(el)

    lat, lon, alt = projected_beam(site_lat, site_lon, az, el, proj_alt=alt)
    return lat[::-1], lon[::-1]


def get_mag_sites():
    filename = os.path.join(os.path.dirname(__file__), 'site_data', 'SuperMAG_sites.txt')
    lon, lat = np.loadtxt(filename, skiprows=44, usecols=(1,2), unpack=True)
    return lat, lon


def generate_sd_fov(radar):

    hdw_data = pydarn.read_hdw_file(radar)
    gate_lat, gate_lon = pydarn.Coords.GEOGRAPHIC(pydarn.RadarID(hdw_data.stid))

    lat = np.concatenate((gate_lat[0,:],gate_lat[:,-1],gate_lat[-1,::-1],gate_lat[::-1,0]))
    lon = np.concatenate((gate_lon[0,:],gate_lon[:,-1],gate_lon[-1,::-1],gate_lon[::-1,0]))
    return lat, lon



def ASI(ax, glat=None, glon=None, elev=None, alt=None, **plotting_params):

    fov_lat, fov_lon = generate_asi_fov(glat, glon, elev, alt)

    if 'fill' in plotting_params:
        fill_params = plotting_params['fill']
        if 'color' not in fill_params:
            fill_params['color'] = plotting_params['color']
        ax.fill(fov_lon, fov_lat, transform=ccrs.Geodetic(), **fill_params)

    nonplotkw = ['name','sites','fill']
    for kw in nonplotkw:
        if kw in plotting_params:
            del plotting_params[kw]

    ax.plot(fov_lon, fov_lat, transform=ccrs.Geodetic(), **plotting_params)



def FPI(ax, glat=None, glon=None, elev=None, alt=None, **plotting_params):

    fov_lat, fov_lon = generate_fpi_beams(glat, glon, elev, alt)

    nonplotkw = ['name','sites']
    for kw in nonplotkw:
        del plotting_params[kw]

    ax.scatter(fov_lon, fov_lat, transform=ccrs.Geodetic(), **plotting_params)


def SDI(ax, glat=None, glon=None, elev=None, alt=None, **plotting_params):

    fov_lat, fov_lon = generate_asi_fov(glat, glon, elev, alt)

    if 'fill' in plotting_params:
        fill_params = plotting_params['fill']
        if 'color' not in fill_params:
            fill_params['color'] = plotting_params['color']
        ax.fill(fov_lon, fov_lat, transform=ccrs.Geodetic(), **fill_params)

    nonplotkw = ['name','sites','fill']
    for kw in nonplotkw:
        if kw in plotting_params:
            del plotting_params[kw]

    ax.plot(fov_lon, fov_lat, transform=ccrs.Geodetic(), **plotting_params)



# Plot ISRs
def ISR(ax, glat=None, glon=None, elev=None, alt=None, **plotting_params):
    fov_lat, fov_lon = generate_asi_fov(glat, glon, elev, alt)

    if 'fill' in plotting_params:
        fill_params = plotting_params['fill']
        if 'color' not in fill_params:
            fill_params['color'] = plotting_params['color']
        ax.fill(fov_lon, fov_lat, transform=ccrs.Geodetic(), **fill_params)

    nonplotkw = ['name', 'fill']
    for kw in nonplotkw:
        if kw in plotting_params:
            del plotting_params[kw]

    ax.plot(fov_lon, fov_lat, transform=ccrs.Geodetic(), **plotting_params)


# Plot AMISRs
def AMISR(ax, glat=None, glon=None, alt=None, radar=None, **plotting_params):
#for site in instruments['AMISR']:
    fov_lat, fov_lon = generate_amisr_fov(glat, glon, alt, radar)

    if 'fill' in plotting_params:
        fill_params = plotting_params['fill']
        if 'color' not in fill_params:
            fill_params['color'] = plotting_params['color']
        ax.fill(fov_lon, fov_lat, transform=ccrs.Geodetic(), **fill_params)

    nonplotkw = ['name','fill']
    for kw in nonplotkw:
        if kw in plotting_params:
            del plotting_params[kw]

    ax.plot(fov_lon, fov_lat, transform=ccrs.Geodetic(), **plotting_params)

# Plot SuperMAG
def SuperMAG(ax, **plotting_params):
    fov_lat, fov_lon = get_mag_sites()

    nonplotkw = ['name']
    for kw in nonplotkw:
        del plotting_params[kw]

    ax.scatter(fov_lon, fov_lat, transform=ccrs.Geodetic(), **plotting_params)

# Plot SuperDARN
def SuperDARN(ax, name=None, **plotting_params):
    fov_lat, fov_lon = generate_sd_fov(name)

    if 'fill' in plotting_params:
        fill_params = plotting_params['fill']
        if 'color' not in fill_params:
            fill_params['color'] = plotting_params['color']
        ax.fill(fov_lon, fov_lat, transform=ccrs.Geodetic(), **fill_params)

    nonplotkw = ['sites','fill']
    for kw in nonplotkw:
        if kw in plotting_params:
            del plotting_params[kw]

    ax.plot(fov_lon, fov_lat, transform=ccrs.Geodetic(), **plotting_params)
