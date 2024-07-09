# mango_map.py
# map MANGO-NATION network, consisting of redline and greenline ASIs and FPIs

import numpy as np
import pymap3d as pm
import matplotlib.pyplot as plt
import cartopy.crs as ccrs
import cartopy.feature as cfeature

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

# Define ASI Networks
asi_groups = [
    {'name': 'MANGO airglow imaging Network',
     'elev': 15.,
     'alt' : 250.,
     'color': 'orange',
     'sites': [{'name': 'A' , 'glat': 43.27,  'glon': -120.35},
               {'name': 'B' , 'glat': 38.15,  'glon': -111.18},
               {'name': 'C' , 'glat': 41.88,  'glon':  -91.50},
               {'name': 'D' , 'glat': 35.20,  'glon':  -82.87},
               {'name': 'E' , 'glat': 48.15,  'glon':  -97.66}]
    },
    {'name': 'MANGO airglow greenline',
     'elev': 15.,
     'alt' : 95.,
     'color': 'lightgreen',
     'sites': [{'name': 'A' , 'glat': 43.27,  'glon': -120.35},
               {'name': 'B' , 'glat': 38.15,  'glon': -111.18},
               {'name': 'C' , 'glat': 41.60,  'glon': -111.60},
               {'name': 'D' , 'glat': 35.20,  'glon': -111.66},
               {'name': 'E' , 'glat': 48.25,  'glon': -117.12},
               {'name': 'F' , 'glat': 31.23,  'glon':  -98.30},
               {'name': 'G' , 'glat': 33.96,  'glon': -107.18}]
    },
    {'name': 'Former MANGO imagers',
    'type': 'ASI',
    'elev': 15.,
    'alt' : 250.,
    'color': 'darkgrey',
    'sites': [{'name': 'A' , 'glat': 40.80,  'glon': -121.46},
              {'name': 'A' , 'glat': 38.11,  'glon':  -96.09},
              {'name': 'A' , 'glat': 33.29,  'glon':  -89.38},
              {'name': 'A' , 'glat': 45.34,  'glon': -108.91}]
    }
]

# Define FPI Networks
fpi_groups = [
    {'name': 'NATION FPI redline',
     'elev': 45.,
     'alt' : 250.,
     'color': 'orange',
     'sites': [{'name': 'A', 'glat': 43.27,  'glon': -120.35},
               {'name': 'A', 'glat': 41.6 ,  'glon': -111.60},
               {'name': 'A', 'glat': 35.20,  'glon': -111.66},
               {'name': 'A', 'glat': 40.16,  'glon':  -88.16}]
    },
    {'name': 'NATION FPI greenline',
     'elev': 45.,
     'alt' : 95.,
     'color': 'lightgreen',
     'sites': [{'name': 'A', 'glat': 43.27,  'glon': -120.35},
               {'name': 'A', 'glat': 41.60,  'glon': -111.6 },
               {'name': 'A', 'glat': 35.20,  'glon': -111.66}]
    }
]


def projected_beam(lat0, lon0, az, el, proj_alt=300.):

    x, y, z = pm.geodetic2ecef(lat0, lon0, 0.)
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




# Use the site locations to find an apprpriate center point
glat_list = list()
glon_list = list()
for group in asi_groups:
    for site in group['sites']:
        glat_list.append(site['glat'])
        glon_list.append(site['glon'])
for group in fpi_groups:
    for site in group['sites']:
        glat_list.append(site['glat'])
        glon_list.append(site['glon'])
cent_glat = (min(glat_list) + max(glat_list))/2.
cent_glon = (min(glon_list) + max(glon_list))/2.

# Set up figure
fig = plt.figure(figsize=(10,10))
proj = ccrs.AzimuthalEquidistant(central_longitude=cent_glon, central_latitude=cent_glat)
ax = plt.subplot(111, projection=proj)
ax.coastlines(resolution='50m',zorder=2)
ax.gridlines()

# Add ASI networks to plot
for network in asi_groups:
    for site in network['sites']:
        fov_lat, fov_lon = generate_asi_fov(site['glat'], site['glon'], network['elev'], network['alt'])
        ax.plot(fov_lon, fov_lat, color=network['color'], label=network['name'], linewidth=3, zorder=6.5, transform=ccrs.Geodetic())

# Add FPI networks to plot
for network in fpi_groups:
    for site in network['sites']:
        fov_lat, fov_lon = generate_fpi_beams(site['glat'], site['glon'], network['elev'], network['alt'])
        ax.scatter(fov_lon, fov_lat, color=network['color'], label=network['name'], linewidth=3, s=50, zorder=7, transform=ccrs.Geodetic())

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

