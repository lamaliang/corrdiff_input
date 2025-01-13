import os
import dask.array as da
import numpy as np
import pandas as pd
import xarray as xr

from util import regrid_dataset, create_and_process_dataarray, is_local_testing

ERA5_CHANNELS = [
    {'name': 'tp', 'variable': 'precitipation'},
    # 500
    {'name': 'z', 'pressure': 500, 'variable': 'geopotential_height'},
    {'name': 't', 'pressure': 500, 'variable': 'temperature'},
    {'name': 'u', 'pressure': 500, 'variable': 'eastward_wind'},
    {'name': 'v', 'pressure': 500, 'variable': 'northward_wind'},
    # 700
    {'name': 'z', 'pressure': 700, 'variable': 'geopotential_height'},
    {'name': 't', 'pressure': 700, 'variable': 'temperature'},
    {'name': 'u', 'pressure': 700, 'variable': 'eastward_wind'},
    {'name': 'v', 'pressure': 700, 'variable': 'northward_wind'},
    # 850
    {'name': 'z', 'pressure': 850, 'variable': 'geopotential_height'},
    {'name': 't', 'pressure': 850, 'variable': 'temperature'},
    {'name': 'u', 'pressure': 850, 'variable': 'eastward_wind'},
    {'name': 'v', 'pressure': 850, 'variable': 'northward_wind'},
    {'name': 'w', 'pressure': 850, 'variable': 'vertical_velocity'}, # W for 850 only
    # 925
    {'name': 'z', 'pressure': 925, 'variable': 'geopotential_height'},
    {'name': 't', 'pressure': 925, 'variable': 'temperature'},
    {'name': 'u', 'pressure': 925, 'variable': 'eastward_wind'},
    {'name': 'v', 'pressure': 925, 'variable': 'northward_wind'},
    # Remaining surface channels
    {'name': 't2m', 'variable': 'temperature_2m'},
    {'name': 'u10', 'variable': 'eastward_wind_10m'},
    {'name': 'v10', 'variable': 'northward_wind_10m'},
    {'name': 'msl', 'variable': 'mean_sea_level_pressure'},
    # Orography channel
    {'name': 'oro', 'variable': 'terrain_height'},
]

def get_prs_paths(folder, subfolder, variables, start_date, end_date):
    date_range = pd.date_range(start=start_date, end=end_date, freq="MS").strftime("%Y%m").tolist()
    if is_local_testing():
        return [
            os.path.join(folder, f"ERA5_PRS_{var}_{yyyymm}_r1440x721_day.nc")
            for var in variables for yyyymm in date_range
        ]

    return [
        os.path.join(
            folder, "PRS", subfolder, var, yyyymm[:4],
            f"ERA5_PRS_{var}_{yyyymm}_r1440x721_day.nc"
        )
        for var in variables for yyyymm in date_range
    ]

def get_sfc_paths(folder, subfolder, variables, start_date, end_date):
    date_range = pd.date_range(start=start_date, end=end_date, freq="MS").strftime("%Y%m").tolist()
    if is_local_testing():
        return [
            os.path.join(folder, f"ERA5_SFC_{var}_201801_r1440x721_day.nc")
            for var in variables
        ]

    return [
        os.path.join(
            folder, "SFC", subfolder, var, yyyymm[:4],
            f"ERA5_SFC_{var}_{yyyymm}_r1440x721_day.nc"
        )
        for var in variables for yyyymm in date_range
    ]

def get_era5_dataset(dir, grid, start_date, end_date):
    pressure_levels = sorted({ch['pressure'] for ch in ERA5_CHANNELS if 'pressure' in ch})
    pressure_level_vars = list(dict.fromkeys(
        ch['name'] for ch in ERA5_CHANNELS if 'pressure' in ch
    ))
    surface_vars = list(dict.fromkeys(
        ch['name'] for ch in ERA5_CHANNELS if 'pressure' not in ch and ch['name'] not in {'oro'}
    ))

    # pressure_level
    duration = slice(str(start_date), str(end_date))
    prs_paths = get_prs_paths(dir, 'day', pressure_level_vars, start_date, end_date)
    era5_prs = xr.open_mfdataset(prs_paths, combine='by_coords').sel(level=pressure_levels, time=duration)

    # surface
    sfc_paths = get_sfc_paths(dir, 'day', surface_vars, start_date, end_date)
    era5_sfc = xr.open_mfdataset(sfc_paths, combine='by_coords').sel(time=duration)
    era5_sfc['tp'] = era5_sfc['tp'] * 24 * 1000
    era5_sfc['tp'].attrs['units'] = 'mm/day' # Convert unit

    # orography
    era5_topo = xr.open_mfdataset(dir + '/ERA5_oro_r1440x721.nc')[['oro']]
    era5_topo = era5_topo.expand_dims(time=era5_sfc.time)
    era5_topo = era5_topo.reindex(time=era5_sfc.time)

    # Merge prs, sfc, topo and rename variables.
    era5 = xr.merge([era5_prs, era5_sfc, era5_topo]).rename({
        ch['name']: ch['variable'] for ch in ERA5_CHANNELS
    })

    # Crop to Taiwan domain given ERA5 is global data.
    lat, lon = grid.XLAT, grid.XLONG
    era5_crop = era5.sel(
        latitude=slice(lat.max().item(), lat.min().item()),
        longitude=slice(lon.min().item(), lon.max().item()))

    # Based on REF grid, regrid TReAD data over spatial dimensions for all timestamps.
    era5_out = regrid_dataset(era5_crop, grid)

    return era5_crop, era5_out

def get_era5(era5_out):
    era5_channel = np.arange(len(ERA5_CHANNELS))
    era5_pressure = [ch.get('pressure', np.nan) for ch in ERA5_CHANNELS]

    # Create channel coordinates
    channel_coords = dict(
        era5_variable=xr.Variable(["era5_channel"], [ch.get('variable') for ch in ERA5_CHANNELS]),
        era5_pressure=xr.Variable(["era5_channel"], era5_pressure),
    )

    # Create ERA5 DataArray
    stack_era5 = da.stack(
        [
            era5_out[ch['variable']].sel(level=ch['pressure']).data
            if 'pressure' in ch else era5_out[ch['variable']].data
            for ch in ERA5_CHANNELS
        ],
        axis=1
    )
    era5_dims = ["time", "era5_channel", "south_north", "west_east"]
    era5_coords = {
        "time": era5_out["time"],
        "era5_channel": era5_channel,
        "south_north": era5_out["south_north"],
        "west_east": era5_out["west_east"],
        "XLAT": era5_out["XLAT"],
        "XLONG": era5_out["XLONG"],
        **channel_coords,
    }
    era5_chunk_sizes = {
        "time": 1,
        "era5_channel": era5_channel.size,
        "south_north": era5_out["south_north"].size,
        "west_east": era5_out["west_east"].size,
    }

    return create_and_process_dataarray("era5", stack_era5, era5_dims, era5_coords, era5_chunk_sizes)

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
    era5_pre_regrid, era5_out = get_era5_dataset(dir, grid, start_date, end_date)
    print(f"\nERA5 dataset =>\n {era5_out}")

    # Generate output fields
    era5 = get_era5(era5_out)
    era5_center = get_era5_center(era5)
    era5_scale = get_era5_scale(era5)
    era5_valid = get_era5_valid(era5)

    return era5, era5_center, era5_scale, era5_valid, era5_pre_regrid, era5_out