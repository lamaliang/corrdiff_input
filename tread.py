import dask.array as da
import numpy as np
import pandas as pd
import xesmf as xe
import xarray as xr

def get_tread_dataset(file, grid, start_date, end_date):
    surface_vars = ['RAINC', 'RAINNC', 'T2', 'U10', 'V10']

    start_datetime = pd.to_datetime(str(start_date), format='%Y%m%d')
    end_datetime = pd.to_datetime(str(end_date), format='%Y%m%d')

    # Read surface level data.
    tread_surface = xr.open_mfdataset(
        file,
        preprocess=lambda ds: ds[surface_vars].assign_coords(
            time=pd.to_datetime(ds['Time'].values.astype(str), format='%Y-%m-%d_%H:%M:%S')
        ).sel(time=slice(start_datetime, end_datetime))
    )

    # Calculate daily mean for T2, U10, and V10. Also sum TP = RAINC+RAINNC and accumulate daily.
    tread = tread_surface[['T2', 'U10', 'V10']].resample(time='1D').mean()
    tread['TP'] = (tread_surface['RAINC'] + tread_surface['RAINNC']).resample(time='1D').sum()

    tread = tread[['TP', 'T2', 'U10', 'V10']].rename({
        "TP": "precipitation",
        "T2": "temperature_2m",
        "U10": "eastward_wind_10m",
        "V10": "northward_wind_10m",
    })

    # Regrid TReAD data over the spatial dimensions for all timestamps, based on CWA coordinates.
    remap = xe.Regridder(tread, grid, method="bilinear")
    # tccip_cwb = tccip_remap(tccip)
    tread_out = xr.concat(
        [remap(tread.isel(time=i)) for i in range(tread.sizes["time"])],
        dim="time"
    )

    # Replace 0 to nan for TReAD domain is smaller than CWB_zarr.
    fill_value = np.nan
    tread_out["temperature_2m"] = tread_out["temperature_2m"].where(tread_out["temperature_2m"] != 0, fill_value)
    tread_out["temperature_2m"].attrs["_FillValue"] = fill_value

    return tread_out

def generate_tread_output(file, grid, start_date, end_date):
    # Extract TReAD data from file.
    tread_out = get_tread_dataset(file, grid, start_date, end_date)

    # cwb_pressure
    cwb_channel = np.arange(4)
    cwb_pressure = xr.DataArray(
        [np.nan, np.nan, np.nan, np.nan],
        dims=["cwb_channel"],
        coords={"cwb_channel": cwb_channel},
        name="cwb_pressure"
    )

    # Define variable names and create DataArray for cwb_variable.
    cwb_var_names = np.array(list(tread_out.data_vars.keys()), dtype="<U26")
    cwb_vars_dask = da.from_array(cwb_var_names, chunks=(4,))

    # cwb_variable
    cwb_variable = xr.DataArray(
        cwb_vars_dask,
        dims=["cwb_channel"],
        coords={"cwb_pressure": cwb_pressure},
        name="cwb_variable"
    )

    # cwb
    stack_tread = da.stack([tread_out[var].data for var in cwb_var_names], axis=1)
    cwb = xr.DataArray(
        stack_tread,
        dims=["time", "cwb_channel", "south_north", "west_east"],
        coords={
            "time": tread_out["time"],
            "cwb_channel": cwb_channel,
            "south_north": tread_out["south_north"],
            "west_east": tread_out["west_east"],
            "XLAT": tread_out["XLAT"],
            "XLONG": tread_out["XLONG"],
            "cwb_pressure": cwb_pressure,
            "cwb_variable": cwb_variable,
        },
        name="cwb"
    )

    # cwb_center
    tread_mean = da.stack(
        [tread_out[var_name].mean(dim=["time", "south_north", "west_east"]).data for var_name in cwb_variable.values],
        axis=0
    )
    cwb_center = xr.DataArray(
        tread_mean,
        dims=["cwb_channel"],
        coords={
            "cwb_pressure": cwb_pressure,
            "cwb_variable": cwb_variable
        },
        name="cwb_center"
    )

    # cwb_scale
    tread_std = da.stack(
        [tread_out[var_name].std(dim=["time", "south_north", "west_east"]).data for var_name in cwb_variable.values],
        axis=0
    )
    cwb_scale = xr.DataArray(
        tread_std,
        dims=["cwb_channel"],
        coords={
            "cwb_pressure": cwb_pressure,
            "cwb_variable": cwb_variable
        },
        name="cwb_center"
    )

    # cwb_valid
    valid = True  
    cwb_valid = xr.DataArray(
        data=da.from_array([valid] * len(tread_out["time"]), chunks=(len(tread_out["time"]),)),
        dims=["time"],
        coords={"time": cwb["time"]},
        name="cwb_valid"
    )

    return cwb, cwb_center, cwb_scale, cwb_valid, cwb_variable
