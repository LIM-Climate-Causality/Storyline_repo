import xarray as xr
import numpy as np
import os
from esmvaltool.diag_scripts.shared import run_diagnostic, get_cfg, group_metadata
import matplotlib.pyplot as plt
from cartopy.util import add_cyclic_point
import cartopy.crs as ccrs
from scipy import stats

def test_mean_significance(sample_data):
    """Test if the mean of the sample_data is significantly different from zero.
    
    Args:
        sample_data (np.ndarray): Array of sample data.
        
    Returns:
        float: p-value from the t-test.
    """
    # Perform a one-sample t-test
    _, p_value = stats.ttest_1samp(sample_data, 0)
    return p_value

def clim_change(target, period1, period2, box=None):
    """
    Calculate the difference in climatological mean between two periods for a specific geographical box.

    Parameters:
    ----------
    target : xarray.DataArray
        The input data array containing the variable of interest (e.g., temperature, precipitation, etc.)
        with dimensions that include 'time', 'lat' (latitude), and 'lon' (longitude).
    period1 : list or tuple of two elements (strings or datetime-like objects)
        Defines the first time period to be averaged, in the form [start_time, end_time].
    period2 : list or tuple of two elements (strings or datetime-like objects)
        Defines the second time period to be averaged, in the form [start_time, end_time].
    box : dict or None, optional
        A dictionary defining the geographical box {'lon_min', 'lon_max', 'lat_min', 'lat_max'}.

    Returns:
    -------
    xarray.DataArray
        A DataArray representing the difference between the climatological means of the two periods for the 
        selected geographical box.
    """
    if box:
        # Select the region based on the box
        target = target.sel(lon=slice(box['lon_min'], box['lon_max'])).sel(lat=slice(box['lat_min'], box['lat_max']))
    
    # Calculate the difference between the two periods
    output = target.sel(time=slice(period2[0], period2[1])).mean(dim='time') - target.sel(time=slice(period1[0], period1[1])).mean(dim='time')
    return output

def plot_function(target_change, p_values, sig_level=0.05):
    """
    Plot target changes and add stippling where p-values indicate significance.

    Parameters:
    ----------
    target_change : xarray.DataArray
        The data array to plot (e.g., difference between two climatological periods).
    p_values : xarray.DataArray
        Data array with p-values, aligned with the dimensions of `target_change`.
    sig_level : float, optional
        Significance level for stippling. Defaults to 0.05 (5%).

    Returns:
    -------
    fig : matplotlib.figure.Figure
        The resulting plot figure with stippling for significance.
    """
    fig, ax = plt.subplots(subplot_kw={'projection': ccrs.PlateCarree()})

    # Add cyclic points for a seamless map plot
    target_change_cyclic, lons_cyclic = add_cyclic_point(target_change, coord=target_change['lon'], axis=1)
    p_values_cyclic, _ = add_cyclic_point(p_values, coord=target_change['lon'], axis=1)

    # Plot the main target change
    cs = ax.contourf(lons_cyclic, target_change['lat'], target_change_cyclic, transform=ccrs.PlateCarree())
    fig.colorbar(cs, ax=ax, orientation='vertical')

    # Create stippling where p-values indicate significance
    sig_mask = p_values_cyclic < sig_level

    # Create the meshgrid for stippling
    stipple_lons, stipple_lats = np.meshgrid(lons_cyclic, target_change['lat'])

    # Ensure sig_mask shape matches stipple_lons and stipple_lats
    if sig_mask.shape != stipple_lons.shape:
        raise ValueError(f"Dimension mismatch: sig_mask shape {sig_mask.shape}, stipple_lons shape {stipple_lons.shape}")

    # Apply the 2D mask to the 2D longitude and latitude arrays
    ax.scatter(stipple_lons[sig_mask], stipple_lats[sig_mask], color='k', s=10, marker='.', alpha=0.5)

    ax.coastlines()
    ax.set_title("Target Change with Significance Stippling")

    return fig

def main(config):
    """Run the diagnostic.""" 
    cfg = get_cfg(os.path.join(config["run_dir"], "settings.yml"))
    print(cfg)
    
    # Define the bounding box for the desired region
    region = {
        'lon_min': 10, 'lon_max': 20,  # Example: Ural region
        'lat_min': 40, 'lat_max': 50
    }

    # Group metadata
    meta_dataset = group_metadata(config["input_data"].values(), "dataset")
    meta = group_metadata(config["input_data"].values(), "alias")
    
    # Create output directories
    output_dir = os.path.join(config["work_dir"], "target_change")
    plot_dir = os.path.join(config["plot_dir"], "target_change")
    os.makedirs(output_dir, exist_ok=True)
    os.makedirs(plot_dir, exist_ok=True)

    ensemble_changes = []

    for dataset, dataset_list in meta_dataset.items():
        model_changes = []
        print(f"Evaluate for {dataset}\n")
        for alias, alias_list in meta.items():
            target_pr = {m["variable_group"]: xr.open_dataset(m["filename"])[m["short_name"]] 
                         for m in alias_list if (m["dataset"] == dataset) and (m["variable_group"] == 'pr')}
            target_gw = {m["variable_group"]: xr.open_dataset(m["filename"])[m["short_name"]]
                         for m in alias_list if (m["dataset"] == dataset) & (m["variable_group"] == 'gw')}
            
            try: 
                target_pr = target_pr['pr'] * 86400  # Convert from kg m-2 s-1 to mm day-1
                target_gw = target_gw['gw']
                print(f"Computing climatological change in model {alias}\n")
                # Calculate precipitation change between the two periods using the specified region
                target_pr_change = clim_change(target_pr, ['1950', '1979'], ['2070', '2099'], box=region)
                target_gw_change = clim_change(target_gw,['1950','1979'],['2070','2099']) 
                model_changes.append(target_pr_change/target_gw_change)
            except KeyError as e:
                print(f"Data for {alias} in {dataset} could not be found: {e}")
                continue
            
        if model_changes:
            model_changes_ar = xr.concat(model_changes, dim='ensemble')
            model_mean = model_changes_ar.mean(dim='ensemble')
            model_maps_pval = xr.apply_ufunc(test_mean_significance, model_changes_ar, input_core_dims=[["ensemble"]],
                                             output_core_dims=[[]], vectorize=True, dask="parallelized")
            ensemble_changes.append(model_mean)
            fig = plot_function(model_mean, model_maps_pval)
            fig.savefig(os.path.join(plot_dir, f"{dataset}_precip_change.png"))

    if ensemble_changes:
        ensemble_changes_ar = xr.concat(ensemble_changes, dim='model')
        ensemble_mean = ensemble_changes_ar.mean(dim='model')
        ensemble_maps_pval = xr.apply_ufunc(test_mean_significance, ensemble_changes_ar, input_core_dims=[["model"]],
                                            output_core_dims=[[]], vectorize=True, dask="parallelized")
        fig = plot_function(ensemble_mean, ensemble_maps_pval)
        fig.savefig(os.path.join(plot_dir, "ensemble_mean_precip_change.png"))

if __name__ == "__main__":
    with run_diagnostic() as config:
        main(config)
