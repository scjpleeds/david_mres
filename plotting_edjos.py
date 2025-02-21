import cartopy.crs as ccrs 
import matplotlib.pyplot as plt
import matplotlib.path as mpath
import numpy as np
import xarray as xr
import cartopy.mpl.ticker as ctk
import math
import matplotlib

matplotlib.use("TkAgg") 

def low_pass_weights(window, cutoff):
    """
    Calculate weights for a low-pass Lanczos filter.

    Parameters:
    -----------
    window : int
        The length of the filter window.
    cutoff : float
        The cutoff frequency in inverse time steps.

    Returns:
    --------
    np.array
        Weights for the filter.
    """
    order = ((window-1)//2)+1
    nwts = 2*order+1 
    w = np.zeros([nwts])
    n = nwts//2
    w[n] = 2*cutoff
    k = np.arange(1., n)
    sigma = np.sin(np.pi*k/n) * n / (np.pi*k)
    firstfactor = np.sin(2.*np.pi*cutoff*k) / (np.pi*k)
    w[n-1:0:-1] = firstfactor * sigma
    w[n+1:-1] = firstfactor * sigma
    return w[1:-1]

def low_pass_filter(da, window, days):
    """
    Apply a low-pass Lanczos filter to an xarray.DataArray.

    Parameters:
    -----------
    da : xarray.DataArray
        The input data array with a 'time' dimension.
    window : int
        The number of time steps in the rolling window.
    days : float
        The cutoff frequency in inverse time steps.

    Returns:
    --------
    xarray.DataArray
        The filtered data.
    """
    # Get the filter weights
    weights = low_pass_weights(window, 1.0 / days)

    # Apply rolling weighted mean
    da_filtered = da.rolling(time=len(weights), center=True).construct("window_dim") \
                   .dot(xr.DataArray(weights, dims=["window_dim"]))

    return da_filtered

# This script will plot EDJO data for ERA5
# Load datasets
edjo_data = xr.open_dataset('ERA5_19592020_EDJO_Ucrit8_60_0_lm.nc')
flood_data = xr.open_dataset('ERA5_19592020_EDJO_Ucrit8_60_0_flood_mask.nc')

# Load wind data
# 
U = xr.open_dataset('/Users/admin/OneDrive - University of Leeds/ARC4/ERA/U19592020/U.nc').sel(level=850, drop=True)
U = U.sel(longitude=slice(-60, 0), latitude=slice(75, 15))
U = low_pass_filter(U['u'],61,10).dropna(dim='time')
#
# The code above here is the necessary preprocessing the get the right domain, level and filtering to match the edjo_data for plotting that I use. The functions 
# for the filtering are in the script. If you want to use these just uncomment and change the filename to your path. The important part is that the time 
# points between the wind field and the edjo and flood data have to match.

# U = xr.open_dataset('your_file_name_here.nc')

# Select the months you want to plot, you can use this snippet anywhere to select the data you need. 
# You could change months to years or days or any other time period.
months = [11,12,1,2,3] #NDJFM 
U = U.sel(time=U['time.month'].isin(months))
edjo_data = edjo_data.sel(time=edjo_data['time.month'].isin(months))
flood_data = flood_data.sel(time=flood_data['time.month'].isin(months))


##############################################################################################################################
######### ------------------------------------------------ Plotting ------------------------------------------------ #########
##############################################################################################################################

# Define lon/lat boundaries
lon_plot, lat_plot = U['longitude'].data, U['latitude'].data
lon1, lon2, lat1, lat2 = [min(lon_plot), max(lon_plot), min(lat_plot), max(lat_plot)]
rect = mpath.Path([[lon1, lat1], [lon2, lat1], [lon2, lat2], [lon1, lat2], [lon1, lat1]]).interpolated(50)
proj = ccrs.AlbersEqualArea(central_longitude=(lon1+lon2)*0.5, central_latitude=(lat1+lat2)*0.5)

# Select days. Here i am using just the index but you could select the dates you want to plot with xarray. 
# You can more days to the list and the loop will plot each one.

days = [500]
# Feel free to modify this to how you want it too look
for day in days: 
    fig, ax = plt.subplots(1, 1, figsize=(15, 15), subplot_kw={'projection': proj})
    ax.coastlines(resolution='50m')

    ax.set_extent([lon1, lon2 + 0.005, lat1 - 0.005, lat2], crs=ccrs.PlateCarree())

    proj_to_data = ccrs.PlateCarree()._as_mpl_transform(ax) - ax.transData
    rect_in_target = proj_to_data.transform_path(rect)
    ax.set_boundary(rect_in_target)
   
    color_val = 12
    clevels = np.linspace(-color_val - 9, color_val + 9, 41) # this sets the contour levels for the wind field. Can change if you wish these are just values I found worked best
    ax.contourf(lon_plot, lat_plot, U.isel(time=day), transform=ccrs.PlateCarree(), levels=clevels, cmap='RdBu_r', extend='both')

    ax.coastlines(resolution='50m')
    gl = ax.gridlines(draw_labels=True, linestyle='dashed', crs=ccrs.PlateCarree())
    gl.top_labels = False
    gl.right_labels = True
    gl.rotate_labels = False
    gl.xlocator = ctk.LongitudeLocator(4)
    gl.ylocator = ctk.LatitudeLocator(6)
    gl.xformatter = ctk.LongitudeFormatter(zero_direction_label=True)
    gl.yformatter = ctk.LatitudeFormatter()
    gl.xlabel_style={'size':16}
    gl.ylabel_style={'size':16}

    # Skip if no EDJ region
    if edjo_data['labels'].isel(time=day).item() == 0:
        continue
    
    # Plot EDJ characteristics
    for n in range(edjo_data['labels'].isel(time=day).item()): 
        x0 = float(edjo_data['lambdabar'].isel(time=day))
        y0 = float(edjo_data['phibar'].isel(time=day))
        orientation = np.pi/2 - np.radians(float(edjo_data['alpha'].isel(time=day))) # This is done so the axis points in the direction of flow. 
        major_axis_length = float(edjo_data['major_axis_length'].isel(time=day))
        minor_axis_length = float(edjo_data['minor_axis_length'].isel(time=day))


        x1 = x0 - math.cos(orientation) * 0.5 * minor_axis_length/111.32# conversion to lat/lon degrees from km 
        y1 = y0 + math.sin(orientation) * 0.5 * minor_axis_length/111.32
        x2 = x0 + math.sin(orientation) * 0.5 * major_axis_length/111.32*math.sin(np.radians(y0))
        y2 = y0 + math.cos(orientation) * 0.5 * major_axis_length/111.32*math.sin(np.radians(y0))

        ax.plot((x0, x1), (y0, y1), '-k', linewidth=3, transform=ccrs.PlateCarree())
        ax.plot((x0, x2), (y0, y2), '-k', linewidth=3, transform=ccrs.PlateCarree())
        ax.plot(x0, y0, '.', markersize=24, transform=ccrs.PlateCarree(), label=r'$\overline{\phi}$ = '+str(np.round(y0, 2)), color='cyan')

    # Plot flood mask
    flooded = flood_data['flood'].isel(time=day)
    ax.contour(lon_plot, lat_plot, flooded, levels=0,colors='black', linewidths=4, transform=ccrs.PlateCarree())

    # Annotate EDJ properties, can add more accordingly
    ax.annotate(r'$\overline{\phi}$ = '+str(np.round(edjo_data['phibar'].isel(time=day).data, 1)), xy=(700, 700), xycoords='axes points')
    ax.annotate(r'$\alpha$ = '+str(np.round(edjo_data['alpha'].isel(time=day).data, 1)), xy=(700, 650), xycoords='axes points')

plt.show()

# can add a plt.save() here if you wish.


# Note: this script will only plot the object with the largest Umass, if you want to plot days with multiple objects let me know 
# and I can modify the script for you. 