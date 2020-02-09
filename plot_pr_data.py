import argparse
import numpy as np
import matplotlib.pyplot as plt
import xarray as xr
import cartopy.crs as ccrs
import cmocean

def convert_pr_units(clim):
    """Convert units of precipitation climatology
    
    Args:
      clim (xarray dat array): Precipitation climatology (units of kg/m2/s)
      
    """
    
    clim.data *= 86400
    clim.attrs['units'] = 'mm/day'
    return clim

def create_plot(clim, model, season, gridlines=False, levels=None):
    """Create plot of precipitation climatology
    
    Args:
      clim (data array): Precipitation climatology
      model (str): Model ID
      season (str): Season (DJF, MAM, JJA, SON)
      gridlines (bool): Select whether to plot gridlines
      levels (list): Tick marks on the colorbar  
    """

    if not levels:
        levels = np.arange(0, 13.5, 1.5)
    
    fig = plt.figure(figsize=[12, 5])
    ax = fig.add_subplot(111, projection=ccrs.PlateCarree(central_longitude=0))
    clim.sel(season=season).plot.contourf(
        ax = ax,
        levels = levels,
        extend = 'max',
        transform = ccrs.PlateCarree(),
        cbar_kwargs = {'label':clim.units},
        cmap = cmocean.cm.haline_r
    )
    ax.coastlines()
    if gridlines:
        plt.gca().gridlines()
    title = '%s precipitation climatology (%s)' %(model, season)
    plt.title(title)

def main(inargs):
    dset = xr.open_dataset(inargs.infile)
    
    clim = dset['pr'].groupby('time.season').mean('time', keep_attrs=True)
    clim = convert_pr_units(clim)
    
    create_plot(clim, dset.attrs['model_id'], inargs.season,
                gridlines=inargs.season, levels=inargs.levels)
    plt.savefig(inargs.outfile)

if __name__ == '__main__':
    description = "Plot the precipitation climatology for a given season"
    
    parser = argparse.ArgumentParser(description=description)
    
    parser.add_argument('infile', type=str, help='Input precipitation data file')
    parser.add_argument('season', type=str,
                        choices=['DJF','MAM','JJA','SON'],
                        help='Season to plot')
    
    parser.add_argument('outfile', type=str, help='Output plot file')

    parser.add_argument("--gridlines", action="store_true", default=False,
                        help="Include gridlines on the plot")

    parser.add_argument("--levels", type=float, nargs='*', default=None,
                        help='list of color levels')
    
    args = parser.parse_args()
    
    main(args)
