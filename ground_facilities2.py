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
output_figure = 'AK_map.png'

# Read in sites file which specifies instruments
with open('ak_sites.yaml', 'r') as f:
    instruments = yaml.safe_load(f)

print(instruments.keys())



## Normalize longitude to be between 0 and 360.
#def reg_lon(lon, vmin=0., vmax=360.):
#    lon = lon % (vmax-vmin)
#    if lon < 0:
#        lon = 360. + lon
#    return lon
#
## Use the site locations to find an appropriate center point
#glat_list = list()
#glon_list = list()
#
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
#cent_glat = 50.
#cent_glon = 250.

# Centeral Alaska
cent_glat = 65.5
cent_glon = -147.7


# Set up figure
fig = plt.figure(figsize=(10,10))
proj = ccrs.Orthographic(central_longitude=cent_glon, central_latitude=cent_glat)
ax = plt.subplot(111, projection=proj)
ax.coastlines(resolution='50m',zorder=0.5)
ax.gridlines()
#ax.set_extent([-155, -60, 20, 80], crs=ccrs.PlateCarree())
ax.set_extent([-170, -125, 50, 75], crs=ccrs.PlateCarree())

# Add all instruments from sites.yaml
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

