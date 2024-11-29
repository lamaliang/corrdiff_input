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

def get_era5_dataset():
    era5_dir = data_path["era5_dir"]
    prsvars = ['u', 'v', 't', 'r', 'z']
    sfcvars = ['msl', 'tp', 't2m', 'u10', 'v10']

    duration = slice(str(iStart), str(iLast))
    era5_prs = xr.open_mfdataset(get_era5_prs_paths(era5_dir, "day", prsvars), combine='by_coords').sel(level=pressure_levels, time=duration)
    era5_sfc = xr.open_mfdataset(get_era5_sfc_paths(era5_dir, "day", sfcvars), combine='by_coords').sel(time=duration)
    era5_topo = xr.open_mfdataset(era5_dir + "/ERA5_oro_r1440x721.nc")[['oro']]

    # Convert units.
    era5_sfc['tp'] = era5_sfc['tp'] * 24 * 1000
    era5_sfc['tp'].attrs['units'] = 'mm/day'
    era5_topo = era5_topo.expand_dims(time=era5_sfc.time)
    era5_topo = era5_topo.reindex(time=era5_sfc.time)

    # Merge prs, sfc, topo.
    era5 = xr.merge([era5_prs, era5_sfc, era5_topo])

    # Rename variables.
    era5 = era5.rename({
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
    era5_remap = xe.Regridder(era5, grid_cwa, method="bilinear")
    # era5_cwb = era5_remap(era5)
    era5_cwb = xr.concat(
        [era5_remap(era5.isel(time=i)) for i in range(era5.sizes["time"])],
        dim="time"
    )

    return era5_cwb

def get_tread_dataset():
    tread_file = data_path["tread_file"]
    sfcvars = ['RAINC', 'RAINNC', 'T2', 'U10', 'V10']

    start_datetime = pd.to_datetime(str(iStart), format='%Y%m%d')
    end_datetime = pd.to_datetime(str(iLast), format='%Y%m%d')

    # Read surface level data.
    td_sfc = xr.open_mfdataset(
        tread_file,
        preprocess=lambda ds: ds[sfcvars].assign_coords(
            time=pd.to_datetime(ds['Time'].values.astype(str), format='%Y-%m-%d_%H:%M:%S')
        ).sel(time=slice(start_datetime, end_datetime))
    )

    # Calculate daily mean for T2, U10, and V10. Also sum TP = RAINC+RAINNC and accumulate daily.
    tccip = td_sfc[['T2', 'U10', 'V10']].resample(time='1D').mean()
    tccip['TP'] = (td_sfc['RAINC'] + td_sfc['RAINNC']).resample(time='1D').sum()

    tccip = tccip[['TP', 'T2', 'U10', 'V10']]
    tccip = tccip.rename({
        "TP": "precipitation",
        "T2": "temperature_2m",
        "U10": "eastward_wind_10m",
        "V10": "northward_wind_10m",
    })

    # Regrid TReAD data over the spatial dimensions for all timestamps, based on CWA coordinates.
    tccip_remap = xe.Regridder(tccip, grid_cwa, method="bilinear")
    # tccip_cwb = tccip_remap(tccip)
    tccip_cwb = xr.concat(
        [tccip_remap(tccip.isel(time=i)) for i in range(tccip.sizes["time"])],
        dim="time"
    )

    # Replace 0 to nan for TReAD domain is smaller than CWB_zarr.
    fill_value = np.nan
    tccip_cwb["temperature_2m"] = tccip_cwb["temperature_2m"].where(tccip_cwb["temperature_2m"] != 0, fill_value)
    tccip_cwb["temperature_2m"].attrs["_FillValue"] = fill_value

    return tccip_cwb

def generate_output_dataset(tccip_cwb, era5_cwb, coords_cwa):
    XTIME = np.datetime64("2024-11-26 15:00:00", "ns")
    coords = coords_cwa
    coords["XTIME"] = XTIME

    # CWB

    # cwb_pressure
    cwb_channel = np.arange(4)
    cwb_pressure = xr.DataArray(
        [np.nan, np.nan, np.nan, np.nan],
        dims=["cwb_channel"],
        coords={"cwb_channel": cwb_channel},
        name="cwb_pressure"
    )

    # Define variable names and create DataArray for cwb_variable
    cwb_vnames = np.array(list(tccip_cwb.data_vars.keys()), dtype="<U26")
    cwb_vars_dask = da.from_array(cwb_vnames, chunks=(4,))

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
    stack_cwb = da.stack([tccip_cwb[var].data for var in cwb_vnames], axis=1)
    south_north_coords = tccip_cwb["south_north"]
    west_east_coords = tccip_cwb["west_east"]

    cwb = xr.DataArray(
        stack_cwb,
        dims=["time", "cwb_channel", "south_north", "west_east"],
        coords={
            "time": tccip_cwb["time"],
            "cwb_channel": cwb_channel,
            "south_north": south_north_coords,
            "west_east": west_east_coords,
            "XLAT": tccip_cwb["XLAT"],
            "XLONG": tccip_cwb["XLONG"],
            "XTIME": XTIME,
            "cwb_pressure": cwb_pressure,
            "cwb_variable": cwb_variable,
        },
        name="cwb"
    )

    # cwb_center
    tccip_cwb_mean = da.stack(
        [tccip_cwb[var_name].mean(dim=["time", "south_north", "west_east"]).data for var_name in cwb_variable.values],
        axis=0
    )

    cwb_center = xr.DataArray(
        tccip_cwb_mean,
        dims=["cwb_channel"],
        coords={
            "XTIME": XTIME,
            "cwb_pressure": cwb_pressure,
            "cwb_variable": cwb_variable
        },
        name="cwb_center"
    )

    # cwb_scale
    tccip_cwb_std = da.stack(
        [tccip_cwb[var_name].std(dim=["time", "south_north", "west_east"]).data for var_name in cwb_variable.values],
        axis=0
    )

    cwb_scale = xr.DataArray(
        tccip_cwb_std,
        dims=["cwb_channel"],
        coords={
            "XTIME": tccip_cwb["XTIME"],
            "cwb_pressure": cwb_pressure,
            "cwb_variable": cwb_variable
        },
        name="cwb_center"
    )

    # cwb_valid
    valid = True  
    cwb_valid = xr.DataArray(
        data=da.from_array([valid] * len(tccip_cwb["time"]), chunks=(len(tccip_cwb["time"]),)),
        dims=["time"],
        coords={
            "time": tccip_cwb["time"],
            "XTIME": XTIME
        },
        name="cwb_valid"
    )

    ## ERA5

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
            era5_cwb[var].sel(level=plev).data if "level" in era5_cwb[var].dims else era5_cwb[var].data
            for var, plev in zip(era5_variables_values, era5_pressure_values)
        ],
        axis=1
    )

    # era5
    era5 = xr.DataArray(
        stack_era5,
        dims=["time", "era5_channel", "south_north", "west_east"],
        coords={
            "time": era5_cwb["time"],
            "era5_channel": era5_channel,
            "south_north": era5_cwb["south_north"],
            "west_east": era5_cwb["west_east"],
            "XLAT": era5_cwb["XLAT"],
            "XLONG": era5_cwb["XLONG"],
            "XTIME": era5_cwb["XTIME"],
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
            "time": tccip_cwb.time  # 保留時間
        }
    )

    out = out.assign_coords(cwb_variable=cwb_variable)
    out = out.reset_coords("cwb_channel", drop=True) if "cwb_channel" in out.coords else out
    out.coords["era5_scale"] = ("era5_channel", era5_scale.data)

    # Write cwb & era5 data variables
    out["cwb"] = cwb
    out["cwb_center"] = cwb_center
    out["cwb_scale"] = cwb_scale
    out["cwb_valid"] = cwb_valid

    out["era5"] = era5
    out["era5_center"] = era5_center
    out["era5_valid"] = era5_valid

    out = out.drop_vars(["south_north", "west_east"])

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

tccip_cwb = get_tread_dataset()
era5_cwb = get_era5_dataset()

# Copy coordinates "latitude" and "longitude"
coord_list = ["XLAT", "XLAT_U", "XLAT_V", "XLONG", "XLONG_U", "XLONG_V"]
coords_cwa = { key: cwa.coords[key] for key in coord_list }

out = generate_output_dataset(tccip_cwb, era5_cwb, coords_cwa)

write_to_zarr(f"corrdiff_dataset_{iStart}_{iLast}.zarr", out)