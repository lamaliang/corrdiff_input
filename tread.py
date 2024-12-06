import dask.array as da
import numpy as np
import pandas as pd
import xarray as xr

from util import regrid_dataset

TREAD_CHANNELS_NO_TP = {
    # Baseline
    "T2": "temperature_2m",
    "U10": "eastward_wind_10m",
    "V10": "northward_wind_10m",
    # C1.x
    "UV10": "windspeed_10m",
    "RH2": "relative_humidity_2m",
    "PSFC": "sea_level_pressure",
}
TREAD_CHANNELS = { "TP": "precipitation" } | TREAD_CHANNELS_NO_TP

def get_tread_dataset(file, grid, start_date, end_date):
    channel_keys_no_tp = list(TREAD_CHANNELS_NO_TP.keys())
    surface_vars = ['RAINC', 'RAINNC'] + channel_keys_no_tp

    start_datetime = pd.to_datetime(str(start_date), format='%Y%m%d')
    end_datetime = pd.to_datetime(str(end_date), format='%Y%m%d')

    # Read surface level data.
    tread_surface = xr.open_mfdataset(
        file,
        preprocess=lambda ds: ds[surface_vars].assign_coords(
            time=pd.to_datetime(ds['Time'].values.astype(str), format='%Y-%m-%d_%H:%M:%S')
        ).sel(time=slice(start_datetime, end_datetime))
    )

    # Calculate daily mean for non-TP channels, and sum TP = RAINC+RAINNC & accumulate daily.
    tread = tread_surface[channel_keys_no_tp].resample(time='1D').mean()
    tread['TP'] = (tread_surface['RAINC'] + tread_surface['RAINNC']).resample(time='1D').sum()

    tread = tread[list(TREAD_CHANNELS.keys())].rename(TREAD_CHANNELS)

    # Based on CWA grid, regrid TReAD data over spatial dimensions for all timestamps.
    tread_out = regrid_dataset(tread, grid)

    # Replace 0 to nan for TReAD domain is smaller than CWB_zarr.
    fill_value = np.nan
    tread_out["temperature_2m"] = tread_out["temperature_2m"].where(tread_out["temperature_2m"] != 0, fill_value)
    tread_out["temperature_2m"].attrs["_FillValue"] = fill_value

    return tread_out

def get_cwb_pressure(cwb_channel):
    return xr.DataArray(
        data=da.from_array(
            [np.nan] * len(TREAD_CHANNELS),
            chunks=(len(TREAD_CHANNELS),)
        ),
        dims=["cwb_channel"],
        coords={"cwb_channel": cwb_channel},
        name="cwb_pressure"
    )

def get_cwb_variable(cwb_var_names, cwb_pressure):
    cwb_vars_dask = da.from_array(cwb_var_names, chunks=(len(TREAD_CHANNELS),))
    return xr.DataArray(
        cwb_vars_dask,
        dims=["cwb_channel"],
        coords={"cwb_pressure": cwb_pressure},
        name="cwb_variable"
    )

def get_cwb(tread_out, cwb_var_names, cwb_channel, cwb_pressure, cwb_variable):
    stack_tread = da.stack([tread_out[var].data for var in cwb_var_names], axis=1)
    return xr.DataArray(
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

def get_cwb_center(tread_out, cwb_pressure, cwb_variable):
    tread_mean = da.stack(
        [tread_out[var_name].mean(dim=["time", "south_north", "west_east"]).data for var_name in cwb_variable.values],
        axis=0
    )

    return xr.DataArray(
        tread_mean,
        dims=["cwb_channel"],
        coords={
            "cwb_pressure": cwb_pressure,
            "cwb_variable": cwb_variable
        },
        name="cwb_center"
    )

def get_cwb_scale(tread_out, cwb_pressure, cwb_variable):
    tread_std = da.stack(
        [tread_out[var_name].std(dim=["time", "south_north", "west_east"]).data for var_name in cwb_variable.values],
        axis=0
    )

    return xr.DataArray(
        tread_std,
        dims=["cwb_channel"],
        coords={
            "cwb_pressure": cwb_pressure,
            "cwb_variable": cwb_variable
        },
        name="cwb_center"
    )

def get_cwb_valid(tread_out, cwb):
    valid = True  
    return xr.DataArray(
        data=da.from_array(
                [valid] * len(tread_out["time"]),
                chunks=(len(tread_out["time"]))
            ),
        dims=["time"],
        coords={"time": cwb["time"]},
        name="cwb_valid"
    )

def generate_tread_output(file, grid, start_date, end_date):
    # Extract TReAD data from file.
    tread_out = get_tread_dataset(file, grid, start_date, end_date)

    ## Prep for generation

    cwb_channel = np.arange(len(TREAD_CHANNELS))
    cwb_pressure = get_cwb_pressure(cwb_channel)
    # Define variable names and create DataArray for cwb_variable.
    cwb_var_names = np.array(list(tread_out.data_vars.keys()), dtype="<U26")

    ## Generate output fields

    cwb_variable = get_cwb_variable(cwb_var_names, cwb_pressure)
    cwb = get_cwb(tread_out, cwb_var_names, cwb_channel, cwb_pressure, cwb_variable)

    cwb_center = get_cwb_center(tread_out, cwb_pressure, cwb_variable)
    cwb_scale = get_cwb_scale(tread_out, cwb_pressure, cwb_variable)
    cwb_valid = get_cwb_valid(tread_out, cwb)

    return cwb, cwb_variable, cwb_center, cwb_scale, cwb_valid
