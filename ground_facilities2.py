# mango_map.py
# map MANGO-NATION network, consisting of redline and greenline ASIs and FPIs

import numpy as np
import pymap3d as pm
import matplotlib.pyplot as plt
import cartopy.crs as ccrs
import cartopy.feature as cfeature
import yaml
import pydarn
import instrumentation2 as instrumentation

# Specify where to find cartopy backgrounds locally
# THIS WILL HAVE TO BE CHANGED FOR EACH USER
# Follow these instructions for how to download and referece these maps
#   http://earthpy.org/tag/cartopy.html
# You may also have to modify the piece of the script that actually adds
# the background map (bottom of this file).
import os
os.environ['CARTOPY_USER_BACKGROUNDS'] = '/Users/e30737/Desktop/Data/cartopy_background'

# Specify output figure file name
output_figure = 'ASI_map.png'

# Define instruments
#   For each new set of instruments you want to add, a dictionary should be
#   added to one of the below lists following the given format.  This specifies
#   the size and altitude of the projected FoV and the color and label that
#   network should recieve on the map.  Currently the "name" field for sites
#   is not practically used in the code and could be set to anything.  It is 
#   there to potentially identify individual sites in the future, or just to 
#   better keep track of which site different coordinates correspond to.

with open('sites.yaml', 'r') as f:
    instruments = yaml.safe_load(f)


print(instruments.keys())


#def projected_beam(lat0, lon0, az, el, proj_alt=300.):
#
#    x, y, z = pm.geodetic2ecef(lat0, lon0, 0.)
#    vx, vy, vz = pm.enu2uvw(np.cos(el)*np.sin(az), np.cos(el)*np.cos(az), np.sin(el), lat0, lon0)
#
#    #earth = pm.Ellipsoid()
#    earth = pm.Ellipsoid.from_name('wgs84')
#    a2 = (earth.semimajor_axis + proj_alt*1000.)**2
#    b2 = (earth.semimajor_axis + proj_alt*1000.)**2
#    c2 = (earth.semiminor_axis + proj_alt*1000.)**2
#
#    A = vx**2/a2 + vy**2/b2 + vz**2/c2
#    B = x*vx/a2 + y*vy/b2 + z*vz/c2
#    C = x**2/a2 + y**2/b2 + z**2/c2 -1
#
#    alpha = (np.sqrt(B**2-A*C)-B)/A
#
#    lat, lon, alt = pm.ecef2geodetic(x + alpha*vx, y + alpha*vy, z + alpha*vz)
#
#    return lat, lon, alt/1000.
#
#
#def generate_asi_fov(site_lat, site_lon, elev, alt, npoint=50):
#    az = np.linspace(0., 360., npoint)*np.pi/180.
#    el = np.full(npoint, elev)*np.pi/180.
#    lat, lon, alt = projected_beam(site_lat, site_lon, az, el, proj_alt=alt)
#    return lat[::-1], lon[::-1]
#
#def generate_fpi_beams(site_lat, site_lon, elev, alt):
#    az = np.arange(0., 360., 90.)*np.pi/180.
#    el = np.full(4, elev)*np.pi/180.
#    lat, lon, alt = projected_beam(site_lat, site_lon, az, el, proj_alt=alt)
#    return lat[::-1], lon[::-1]
#
#
#def generate_amisr_fov(site_lat, site_lon, radar):
#    filename = os.path.join(os.path.dirname(__file__), 'site_data', '{}GratingLimits.txt'.format(radar.replace('-','').lower()))
#    az, el = np.loadtxt(filename, usecols=[0,1], unpack=True)
#    az = np.deg2rad(az)
#    el = np.deg2rad(el)
#    #az = data[:,0]*np.pi/180.
#    #el = data[:,1]*np.pi/180.
#
#    lat, lon, alt = projected_beam(site_lat, site_lon, az, el, proj_alt=450)
#    return lat[::-1], lon[::-1]
#
#
#def get_mag_sites():
#    #self.color = color
#    filename = os.path.join(os.path.dirname(__file__), 'site_data', 'SuperMAG_sites.txt')
#    lon, lat = np.loadtxt(filename, skiprows=44, usecols=(1,2), unpack=True)
#    #self.sites = np.array([data[:,1],data[:,0],np.zeros(data.shape[0])]).T
#    return lat, lon
#
#
#def generate_sd_fov(radar):
#
#    hdw_data = pydarn.read_hdw_file(radar)
#    stid = hdw_data.stid
#    #radar_info = pydarn.SuperDARNRadars.radars
#
#    #if radars:
#    #    self.sites = [SuperDARN(stid) for stid, info in radar_info.items() if info.hardware_info.abbrev in radars]
#    #else:
#    #    self.sites = [SuperDARN(stid) for stid in radar_info.keys()]
#
#    #gate_lat, gate_lon = pydarn.Coords.GEOGRAPHIC(stid)
#    #print(pydarn.RadarID.CLY)
#    gate_lat, gate_lon = pydarn.Coords.GEOGRAPHIC(pydarn.RadarID.CLY)
#    lat = np.concatenate((gate_lat[0,:],gate_lat[:,-1],gate_lat[-1,::-1],gate_lat[::-1,0]))
#    lon = np.concatenate((gate_lon[0,:],gate_lon[:,-1],gate_lon[-1,::-1],gate_lon[::-1,0]))
#    return lat, lon


def reg_lon(lon, vmin=0., vmax=360.):
    lon = lon % (vmax-vmin)
    if lon < 0:
        lon = 360. + lon
    return lon


#def combine_dict(d1, d2, skip=[]):
#    d0 = d1.copy()
#    for k, v in d2.items():
#        if k in skip:
#            continue
#        else:
#            d0[k] = v
#    return d0

#def ASI(ax, network):
#
#    for site in network['sites']:
#        fov_lat, fov_lon = generate_asi_fov(site['glat'], site['glon'], network['elev'], network['alt'])
#        ax.plot(fov_lon, fov_lat, color=network['color'], label=network['name'], linewidth=3, zorder=6.5, transform=ccrs.Geodetic())
#

## Network object
##   - name
##   - plotting color, linestyle, ect
#class Network(object):
#    def __init__(self, name, color):
#        self.name = name
#        self.color = color
#
## Instrument object
##   - name
##   - has it's own plotting attributs unless inherited from network?
##   - site coordinates
##   - FoV arrays
#class ASI(Network):
#    def __init__(self, name, glat, glon):
#        self.glat = glat
#        self.glon = glon
#
#    def gnerate_fov():
#        pass



# Use the site locations to find an appropriate center point
glat_list = list()
glon_list = list()

#for network in instruments['ASI']:
#    for site in network['sites']:
#        glat_list.append(site['glat'])
#        glon_list.append(reg_lon(site['glon']))
#
#for network in instruments['FPI']:
#    for site in network['sites']:
#        glat_list.append(site['glat'])
#        glon_list.append(reg_lon(site['glon']))
#
#cent_glat = (min(glat_list) + max(glat_list))/2.
#cent_glon = (min(glon_list) + max(glon_list))/2.
cent_glat = 42.
cent_glon = 250.
print(cent_glat, cent_glon)

# Set up figure
fig = plt.figure(figsize=(10,10))
proj = ccrs.AzimuthalEquidistant(central_longitude=cent_glon, central_latitude=cent_glat)
ax = plt.subplot(111, projection=proj)
ax.coastlines(resolution='50m',zorder=2)
ax.gridlines()
#ax.set_extent([-125, -70, 20, 55], crs=ccrs.PlateCarree())
#ax.set_extent([-125, -70, 20, 55], crs=ccrs.PlateCarree())

#add_instrument['ASI'] = all_sky

for inst_type, values in instruments.items():
    print(inst_type)

    plot_funct = getattr(instrumentation, inst_type)

    # Set defaults for this instrument type
    if 'default' in values:
        default_params = values['default']
    else:
        default_params = dict()

    # Plot networks
    if 'networks' in values:
        for network in values['networks']:
            print(network['name'])

            params = default_params | network
            params['label'] = params['name']

            for site in network['sites']:
                params = params | site
                plot_funct(ax, **params)
   
    # Plot individual instruments
    if 'sites' in values:
        for site in values['sites']:
            print(site['name'])
            params = default_params | site
            params['label'] = params['name']
            plot_funct(ax, **params)
    
    #plot_funct = getattr(instrumentation, inst)
    #plot_funct(ax, vals)



## Plot ASI networks
#for network in instruments['ASI']:
#    for site in network['sites']:
#        fov_lat, fov_lon = generate_asi_fov(site['glat'], site['glon'], network['elev'], network['alt'])
#        ax.plot(fov_lon, fov_lat, color=network['color'], label=network['name'], linewidth=3, zorder=6.5, transform=ccrs.Geodetic())
#
## Plot FPI networks
#for network in instruments['FPI']:
#    for site in network['sites']:
#        fov_lat, fov_lon = generate_fpi_beams(site['glat'], site['glon'], network['elev'], network['alt'])
#        ax.scatter(fov_lon, fov_lat, color=network['color'], label=network['name'], linewidth=3, s=50, zorder=7, transform=ccrs.Geodetic())
#
## Plot ISRs
#for site in instruments['ISR']:
#    fov_lat, fov_lon = generate_asi_fov(site['glat'], site['glon'], site['elev'], site['alt'])
#    ax.plot(fov_lon, fov_lat, color=site['color'], label=site['name'], linewidth=3, zorder=6.5, transform=ccrs.Geodetic())
#
## Plot AMISRs
#for site in instruments['AMISR']:
#    fov_lat, fov_lon = generate_amisr_fov(site['glat'], site['glon'], site['radar'])
#    ax.plot(fov_lon, fov_lat, color=site['color'], label=site['name'], linewidth=3, zorder=6.5, transform=ccrs.Geodetic())
#
## Plot mag sites
#fov_lat, fov_lon = get_mag_sites()
#ax.scatter(fov_lon, fov_lat, color=instruments['SuperMAG']['color'], transform=ccrs.Geodetic())
#
## Plot SuperDARN sites
#for site in instruments['SuperDARN']['radars']:
#    fov_lat, fov_lon = generate_sd_fov(site)
#    ax.plot(fov_lon, fov_lat, color=instruments['SuperDARN']['color'], transform=ccrs.Geodetic())

# Add legend to plot
handles, labels = ax.get_legend_handles_labels()
unique_handles = list()
unique_labels = list()
for hand, lab in zip(handles, labels):
    if lab not in unique_labels:
        unique_handles.append(hand)
        unique_labels.append(lab)
ax.legend(unique_handles, unique_labels)

# Add background without messing up the map extent
map_extent = ax.get_extent(crs=proj)
ax.background_img(name='BM',resolution='mid')
ax.set_extent(map_extent, crs=proj)

# Save output figure
plt.savefig(output_figure, bbox_inches='tight')

