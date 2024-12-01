import os
import dask.array as da
import numpy as np
import pandas as pd
import xesmf as xe
import xarray as xr

pressure_levels = [1000, 925, 850, 700, 500]

def isLocal(folder) -> bool:
    return "Reanalysis" not in str(folder) # TODO: Remove the hack

def get_prs_paths(folder, subfolder, variables, start_date):
    year = str(start_date)[:4]
    if isLocal(folder):
        return [
            os.path.join(folder, f"./ERA5_PRS_{var}_201801_r1440x721_day.nc")
            for var in variables
        ]

    return [
        os.path.join(
            folder, "PRS", subfolder, var, str(year),
            f"ERA5_PRS_{var}_{year}{month:02d}_r1440x721_day.nc"
        )
        for var in variables for month in range(1, 13)
    ]

def get_sfc_paths(folder, subfolder, variables, start_date):
    year = str(start_date)[:4]
    if isLocal(folder):
        return [
            os.path.join(folder, f"./ERA5_SFC_{var}_201801_r1440x721_day.nc")
            for var in variables
        ]

    return [
        os.path.join(
            folder, "SFC", subfolder, var, str(year),
            f"ERA5_SFC_{var}_{year}{month:02d}_r1440x721_day.nc"
        )
        for var in variables for month in range(1, 13)
    ]

def get_era5_dataset(dir, grid, start_date, end_date):
    pressure_level_vars = ['z', 'u', 'v', 't', 'r']
    surface_vars = ['msl', 'tp', 't2m', 'u10', 'v10']

    # pressure_level
    duration = slice(str(start_date), str(end_date))
    prs_paths = get_prs_paths(dir, "day", pressure_level_vars, start_date)
    era5_prs = xr.open_mfdataset(prs_paths, combine='by_coords').sel(level=pressure_levels, time=duration)

    # surface
    sfc_paths = get_sfc_paths(dir, "day", surface_vars, start_date)
    era5_sfc = xr.open_mfdataset(sfc_paths, combine='by_coords').sel(time=duration)
    era5_sfc['tp'] = era5_sfc['tp'] * 24 * 1000
    era5_sfc['tp'].attrs['units'] = 'mm/day' # Convert unit

    # orography
    era5_topo = xr.open_mfdataset(dir + "/ERA5_oro_r1440x721.nc")[['oro']]
    era5_topo = era5_topo.expand_dims(time=era5_sfc.time)
    era5_topo = era5_topo.reindex(time=era5_sfc.time)

    # Merge prs, sfc, topo and rename variables.
    era5 = xr.merge([era5_prs, era5_sfc, era5_topo]).rename({
        "z": "geopotential_height",
        "u": "eastward_wind",
        "v": "northward_wind",
        "t": "temperature",
        "r": "relative_humidity",
        "msl": "mean_sea_level_pressure",
        "t2m": "temperature_2m",
        "u10": "eastward_wind_10m",
        "v10": "northward_wind_10m",
        "tp" : "precipitation",
        "oro": "terrain_height"
    })

    # Regrid ERA5 data over the spatial dimensions for all timestamps, based on CWA coordinates.
    remap = xe.Regridder(era5, grid, method="bilinear")
    # era5_out = era5_remap(era5)
    era5_out = xr.concat(
        [remap(era5.isel(time=i)) for i in range(era5.sizes["time"])],
        dim="time"
    )

    return era5_out

def get_era5(era5_out, stack_era5, era5_channel, era5_pressure_values, era5_variables_values):
    return xr.DataArray(
        stack_era5,
        dims=["time", "era5_channel", "south_north", "west_east"],
        coords={
            "time": era5_out["time"],
            "era5_channel": era5_channel,
            "south_north": era5_out["south_north"],
            "west_east": era5_out["west_east"],
            "XLAT": era5_out["XLAT"],
            "XLONG": era5_out["XLONG"],
            "era5_pressure": xr.DataArray(era5_pressure_values, dims=["era5_channel"], coords={"era5_channel": era5_channel}),
            "era5_variable": xr.DataArray(era5_variables_values, dims=["era5_channel"], coords={"era5_channel": era5_channel}),
        },
        name="era5"
    )

def get_era5_center(era5):
    era5_mean = da.stack(
        [
            era5.isel(era5_channel=channel).mean(dim=["time", "south_north", "west_east"]).data
            for channel in era5["era5_channel"].values
        ],
        axis=0
    )

    return xr.DataArray(
        era5_mean,
        dims=["era5_channel"],
        coords={
            "era5_pressure": era5["era5_pressure"],
            "era5_variable": era5["era5_variable"]
        },
        name="era5_center"
    )

def get_era5_scale(era5):
    era5_std = da.stack(
        [
            era5.isel(era5_channel=channel).std(dim=["time", "south_north", "west_east"]).data
            for channel in era5["era5_channel"].values
        ],
        axis=0
    )
    return xr.DataArray(
        era5_std,
        dims=["era5_channel"],
        coords={
            "era5_pressure": era5["era5_pressure"],
            "era5_variable": era5["era5_variable"]
        },
        name="era5_scale"
    )
    return

def get_era5_valid(era5):
    valid = True
    return xr.DataArray(
        data=da.from_array(
                [[valid] * len(era5["era5_channel"])] * len(era5["time"]),
                chunks=(len(era5["time"]), len(era5["era5_channel"]))
            ),
        dims=["time", "era5_channel"],
        coords={
            "time": era5["time"],
            "era5_channel": era5["era5_channel"],
            "era5_pressure": era5["era5_pressure"],
            "era5_variable": era5["era5_variable"]
        },
        name="era5_valid"
    )

def generate_era5_output(dir, grid, start_date, end_date):
    # Extract ERA5 data from file.
    era5_out = get_era5_dataset(dir, grid, start_date, end_date)

    ## Prep for generation

    era5_channel = np.arange(31)
    era5_pressure_values = np.repeat(pressure_levels, 5)
    era5_pressure_values = np.append(era5_pressure_values, [np.nan] * 6)
    era5_variables_values = [
            'geopotential_height',
            'eastward_wind',
            'northward_wind',
            'temperature',
            'relative_humidity'
        ] * 5 + [
            'mean_sea_level_pressure',
            'precipitation',
            'temperature_2m',
            'eastward_wind_10m',
            'northward_wind_10m',
            'terrain_height'
        ]

    stack_era5 = da.stack(
        [
            era5_out[var].sel(level=plev).data if "level" in era5_out[var].dims else era5_out[var].data
            for var, plev in zip(era5_variables_values, era5_pressure_values)
        ],
        axis=1
    )

    ## Generate output fields

    era5 = get_era5(era5_out, stack_era5, era5_channel, era5_pressure_values, era5_variables_values)
    era5_center = get_era5_center(era5)
    era5_scale = get_era5_scale(era5)
    era5_valid = get_era5_valid(era5)

    return era5, era5_center, era5_scale, era5_valid