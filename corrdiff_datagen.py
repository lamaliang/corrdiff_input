import os
import zarr
import xarray as xr
import pandas as pd
import numpy as np
import xesmf as xe
import dask.array as da
from dask.diagnostics import ProgressBar

##
# Configuration
##

iStart = 20180101
iLast = 20180103
pressure_levels = [1000, 925, 850, 700, 500]

LOCAL = True
yyyymm = str(iStart)[:6]
data_path = {
    # LOCAL
    "cwa_ref": "./data/cwa_dataset_example.zarr",
    "tread_file": f"./data/wrfo2D_d02_{yyyymm}.nc",
    "era5_dir": "./data",
} if LOCAL else {
    # REMOTE
    "cwa_ref": "/lfs/home/dadm/data/cwa_dataset.zarr",
    "tread_file": f"/lfs/archive/TCCIP_data/TReAD/SFC/hr/wrfo2D_d02_{yyyymm}.nc",
    "era5_dir": "/lfs/archive/Reanalysis/ERA5",
}

##
# Functions
##

def get_era5_prs_paths(folder, subfolder, variables):
    year = str(iStart)[:4]
    return [
        os.path.join(folder, f"./ERA5_PRS_{var}_201801_r1440x721_day.nc")
        for var in variables
    ] if LOCAL else [
        os.path.join(
            folder, "PRS", subfolder, var, str(year),
            f"ERA5_PRS_{var}_{year}{month:02d}_r1440x721_day.nc"
        )
        for var in variables for month in range(1, 13)
    ]

def get_era5_sfc_paths(folder, subfolder, variables):
    year = str(iStart)[:4]
    return [
        os.path.join(folder, f"./ERA5_SFC_{var}_201801_r1440x721_day.nc")
        for var in variables
    ] if LOCAL else [
        os.path.join(
            folder, "SFC", subfolder, var, str(year),
            f"ERA5_SFC_{var}_{year}{month:02d}_r1440x721_day.nc"
        )
        for var in variables for month in range(1, 13)
    ]

def get_tread_dataset(tread_file):
    surface_vars = ['RAINC', 'RAINNC', 'T2', 'U10', 'V10']

    start_datetime = pd.to_datetime(str(iStart), format='%Y%m%d')
    end_datetime = pd.to_datetime(str(iLast), format='%Y%m%d')

    # Read surface level data.
    tread_surface = xr.open_mfdataset(
        tread_file,
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
    remap = xe.Regridder(tread, grid_cwa, method="bilinear")
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

def get_era5_dataset(era5_dir):
    pressure_level_vars = ['u', 'v', 't', 'r', 'z']
    surfact_vars = ['msl', 'tp', 't2m', 'u10', 'v10']

    duration = slice(str(iStart), str(iLast))
    era5_prs = xr.open_mfdataset(get_era5_prs_paths(era5_dir, "day", pressure_level_vars), combine='by_coords').sel(level=pressure_levels, time=duration)
    era5_sfc = xr.open_mfdataset(get_era5_sfc_paths(era5_dir, "day", surfact_vars), combine='by_coords').sel(time=duration)
    era5_topo = xr.open_mfdataset(era5_dir + "/ERA5_oro_r1440x721.nc")[['oro']]

    # Convert units.
    era5_sfc['tp'] = era5_sfc['tp'] * 24 * 1000
    era5_sfc['tp'].attrs['units'] = 'mm/day'
    era5_topo = era5_topo.expand_dims(time=era5_sfc.time)
    era5_topo = era5_topo.reindex(time=era5_sfc.time)

    # Merge prs, sfc, topo and rename variables.
    era5 = xr.merge([era5_prs, era5_sfc, era5_topo]).rename({
        "u": "eastward_wind",
        "v": "northward_wind",
        "t": "temperature",
        "r": "relative_humidity",
        "z": "geopotential_height",
        "msl": "mean_sea_level_pressure",
        "t2m": "temperature_2m",
        "u10": "eastward_wind_10m",
        "v10": "northward_wind_10m",
        "tp" : "precipitation",
        "oro": "terrain_height"
    })

    # Regrid ERA5 data over the spatial dimensions for all timestamps, based on CWA coordinates.
    remap = xe.Regridder(era5, grid_cwa, method="bilinear")
    # era5_out = era5_remap(era5)
    era5_out = xr.concat(
        [remap(era5.isel(time=i)) for i in range(era5.sizes["time"])],
        dim="time"
    )

    return era5_out

def generate_tread_output(tread_out, XTIME):
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
        coords={
            "XTIME": ("cwb_channel", [XTIME] * len(cwb_channel)),
            "cwb_pressure": cwb_pressure
        },
        name="cwb_variable"
    )

    # cwb
    stack_tread = da.stack([tread_out[var].data for var in cwb_var_names], axis=1)
    south_north_coords = tread_out["south_north"]
    west_east_coords = tread_out["west_east"]

    cwb = xr.DataArray(
        stack_tread,
        dims=["time", "cwb_channel", "south_north", "west_east"],
        coords={
            "time": tread_out["time"],
            "cwb_channel": cwb_channel,
            "south_north": south_north_coords,
            "west_east": west_east_coords,
            "XLAT": tread_out["XLAT"],
            "XLONG": tread_out["XLONG"],
            "XTIME": XTIME,
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
            "XTIME": XTIME,
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
            "XTIME": tread_out["XTIME"],
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
        coords={
            "time": tread_out["time"],
            "XTIME": XTIME
        },
        name="cwb_valid"
    )

    return cwb, cwb_center, cwb_scale, cwb_valid, cwb_variable

def generate_era5_output(era5_out):
    era5_channel = np.arange(31)
    era5_pressure_values = np.tile(pressure_levels, 5) 
    era5_pressure_values = np.append(era5_pressure_values, [np.nan] * 6) 

    era5_variables_values = [
        'eastward_wind', 'northward_wind', 'temperature', 'relative_humidity', 'geopotential_height',
        'eastward_wind', 'northward_wind', 'temperature', 'relative_humidity', 'geopotential_height',
        'eastward_wind', 'northward_wind', 'temperature', 'relative_humidity', 'geopotential_height',
        'eastward_wind', 'northward_wind', 'temperature', 'relative_humidity', 'geopotential_height',
        'eastward_wind', 'northward_wind', 'temperature', 'relative_humidity', 'geopotential_height',
        'mean_sea_level_pressure', 'precipitation', 'temperature_2m', 'eastward_wind_10m', 'northward_wind_10m', 'terrain_height'
    ]

    stack_era5 = da.stack(
        [
            era5_out[var].sel(level=plev).data if "level" in era5_out[var].dims else era5_out[var].data
            for var, plev in zip(era5_variables_values, era5_pressure_values)
        ],
        axis=1
    )

    # era5
    era5 = xr.DataArray(
        stack_era5,
        dims=["time", "era5_channel", "south_north", "west_east"],
        coords={
            "time": era5_out["time"],
            "era5_channel": era5_channel,
            "south_north": era5_out["south_north"],
            "west_east": era5_out["west_east"],
            "XLAT": era5_out["XLAT"],
            "XLONG": era5_out["XLONG"],
            "XTIME": era5_out["XTIME"],
            "era5_pressure": xr.DataArray(era5_pressure_values, dims=["era5_channel"], coords={"era5_channel": era5_channel}),
            "era5_variable": xr.DataArray(era5_variables_values, dims=["era5_channel"], coords={"era5_channel": era5_channel}),
        },
        name="era5"
    )

    # era5_center
    era5_mean = da.stack(
        [
            era5.isel(era5_channel=channel).mean(dim=["time", "south_north", "west_east"]).data
            for channel in era5["era5_channel"].values
        ],
        axis=0
    )
    era5_center = xr.DataArray(
        era5_mean,
        dims=["era5_channel"],
        coords={
            "XTIME": era5["XTIME"],
            "era5_pressure": era5["era5_pressure"],
            "era5_variable": era5["era5_variable"]
        },
        name="era5_center"
    )

    # era5_scale
    era5_std = da.stack(
        [
            era5.isel(era5_channel=channel).std(dim=["time", "south_north", "west_east"]).data
            for channel in era5["era5_channel"].values
        ],
        axis=0
    )
    era5_scale = xr.DataArray(
        era5_std,
        dims=["era5_channel"],
        coords={
            "XTIME": era5["XTIME"],
            "era5_pressure": era5["era5_pressure"],
            "era5_variable": era5["era5_variable"]
        },
        name="era5_scale"
    )

    # era5_valid
    era5_valid = xr.DataArray(
        data=True,
        dims=["time", "era5_channel"],
        coords={
            "time": era5["time"],
            "era5_channel": era5["era5_channel"],
            "XTIME": era5["XTIME"],
            "era5_pressure": era5["era5_pressure"],
            "era5_variable": era5["era5_variable"]
        },
        name="era5_valid"
    )

    return era5, era5_center, era5_scale, era5_valid

def generate_output_dataset(tread_out, era5_out, coords_cwa):
    XTIME = np.datetime64("2024-11-26 15:00:00", "ns")
    coords = coords_cwa
    coords["XTIME"] = XTIME

    # Generate CWB (i.e., TReAD) and ERA5 output fields.
    cwb, cwb_center, cwb_scale, cwb_valid, cwb_variable = generate_tread_output(tread_out, XTIME)
    era5, era5_center, era5_scale, era5_valid = generate_era5_output(era5_out)

    # Copy cwb coordinates and write new cwb and era5 coordinates; also replace XTIME with new coords.
    coords = {
        key: value.drop_vars("XTIME") if isinstance(value, (xr.DataArray, xr.Dataset)) and "XTIME" in value.coords else value
        for key, value in coords.items()
    }

    out = xr.Dataset(
        coords={
            "XLONG": coords["XLONG"],
            "XLAT": coords["XLAT"],
            "XLONG_U": coords["XLONG_U"],
            "XLAT_U": coords["XLAT_U"],
            "XLONG_V": coords["XLONG_V"],
            "XLAT_V": coords["XLAT_V"],
            "XTIME": XTIME,
            "time": tread_out.time  # 保留時間
        }
    )

    out = out.assign_coords(cwb_variable=cwb_variable)
    out = out.reset_coords("cwb_channel", drop=True) if "cwb_channel" in out.coords else out

    # Write cwb & era5 data
    out["cwb"] = cwb
    out["cwb_center"] = cwb_center
    out["cwb_scale"] = cwb_scale
    out["cwb_valid"] = cwb_valid

    out["era5"] = era5
    out["era5_center"] = era5_center
    out["era5_valid"] = era5_valid
    out.coords["era5_scale"] = ("era5_channel", era5_scale.data)

    out = out.drop_vars(["south_north", "west_east"])
    out = out.drop_vars(["cwb_channel", "era5_channel"])

    return out

def write_to_zarr(out_path, out_ds):
    comp = zarr.Blosc(cname='zstd', clevel=3, shuffle=2)
    encoding = { var: {'compressor': comp} for var in out_ds.data_vars }
    # with ProgressBar():
    #     out.to_zarr(out_path, mode='w', encoding=encoding, compute=True)

    print(f"Data successfully saved to [{out_path}]")


##
# Main
##

# Grid of CorrDiff data as reference
cwa = xr.open_zarr(data_path["cwa_ref"])
grid_cwa = xr.Dataset({ "lat": cwa.XLAT, "lon": cwa.XLONG })

# Get TReAD and ERA5 datasets.
tread_out = get_tread_dataset(data_path["tread_file"])
era5_out = get_era5_dataset(data_path["era5_dir"])

# Copy coordinates "latitude" and "longitude"
coord_list = ["XLAT", "XLAT_U", "XLAT_V", "XLONG", "XLONG_U", "XLONG_V"]
coords_cwa = { key: cwa.coords[key] for key in coord_list }

out = generate_output_dataset(tread_out, era5_out, coords_cwa)

write_to_zarr(f"corrdiff_dataset_{iStart}_{iLast}.zarr", out)