# ground_facilities.py
# map relevant instumentation in northern and southern hemispheres

import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
# from apexpy import Apex
import matplotlib.gridspec as gridspec
import cartopy.crs as ccrs
import cartopy.feature as cfeature

from instrumentation import *

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





def main():
    map()

if __name__ == '__main__':
    main()
