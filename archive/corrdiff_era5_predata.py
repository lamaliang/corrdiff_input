import xarray as xr
import pandas as pd
import numpy as np
import os
import dask.array as da
import zarr
from dask.diagnostics import ProgressBar
import xesmf as xe



iStrt = 20180101
iLast = 20180111
yr = str(iStrt)[:4]
time = slice(str(iStrt),str(iLast))
yyyymm = str(iStrt)[:6]
plevs = [1000, 925, 850, 700, 500]


# CWA Reference data
cwa = xr.open_zarr('./data/cwa_dataset_example.zarr')
lat_cwa = cwa.XLAT
lon_cwa = cwa.XLONG
grid_cwa = xr.Dataset({"lat": lat_cwa,"lon": lon_cwa,})


# === TCCIP TReAD data ===
ifnS = f"./wrfo2D_d02_{yyyymm}.nc"
sfcvars = ['RAINC','RAINNC','T2','U10','V10']

stime = pd.to_datetime(str(iStrt), format='%Y%m%d')
etime = pd.to_datetime(str(iLast), format='%Y%m%d')


# --- Read Surface level data ---
td_sfc = xr.open_mfdataset(
    ifnS,
    preprocess=lambda ds: ds[sfcvars].assign_coords(
        time=pd.to_datetime(ds['Time'].values.astype(str), format='%Y-%m-%d_%H:%M:%S')
    ).sel(time=slice(stime, etime))
)


# --- Calculus daily mean for T2,U10,V10 ; TP = RAINC+RAINNC and calculus daily accumulation ---
tccip = td_sfc[['T2', 'U10', 'V10']].resample(time='1D').mean()
tccip['TP'] = (td_sfc['RAINC'] + td_sfc['RAINNC']).resample(time='1D').sum()
tccip = tccip[['TP', 'T2', 'U10', 'V10']]

variable_mapping = {
    "TP": "precipitation",
    "T2": "temperature_2m",
    "U10": "eastward_wind_10m",
    "V10": "northward_wind_10m",
}


tccip = tccip.rename(variable_mapping)


# --- Regridding to CWA coordinate ---
tccip_remap = xe.Regridder(tccip, grid_cwa, method="bilinear")
tccip_cwb = tccip_remap(tccip)

# replace 0 to nan due to TReAD domain is smaller than CWB_Zarr
fill_value = np.nan  

tccip_cwb["temperature_2m"] = tccip_cwb["temperature_2m"].where(tccip_cwb["temperature_2m"] != 0, fill_value)
tccip_cwb["temperature_2m"].attrs["_FillValue"] = fill_value


# === ERA5 data ===
indir = './data'
prsvars = ['u','v','t','r','z']
sfcvars = ['msl','tp','t2m','u10','v10']

def get_file_prs_paths(variables, subfolder):
    return [
        os.path.join(
            indir, "PRS", subfolder, var, str(yr),
            f"ERA5_PRS_{var}_{yr}{month:02d}_r1440x721_day.nc"
        )
        for var in variables for month in range(1, 13)
    ]

def get_file_sfc_paths(variables, subfolder):
    return [
        os.path.join(
            indir, "SFC", subfolder, var, str(yr),
            f"ERA5_SFC_{var}_{yr}{month:02d}_r1440x721_day.nc"
        )
        for var in variables for month in range(1, 13)
    ]


era5_prs = xr.open_mfdataset(get_file_prs_paths(prsvars, "day"),combine='by_coords').sel(level=plevs, time=time)
era5_sfc = xr.open_mfdataset(get_file_sfc_paths(sfcvars, "day"),combine='by_coords',).sel(time=time)
era5_topo = xr.open_mfdataset("./data/ERA5_oro_r1440x721.nc")[['oro']]


# -- units convert ---
era5_sfc['tp'] = era5_sfc['tp'] * 24 * 1000
era5_sfc['tp'].attrs['units'] = 'mm/day'
era5_topo = era5_topo.expand_dims(time=era5_sfc.time)
era5_topo = era5_topo.reindex(time=era5_sfc.time)


# --- Merge prs,sfc,topo ---
era5 = xr.merge([era5_prs,era5_sfc,era5_topo])

# --- Rename ---
variable_mapping = {
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
}

era5 = era5.rename(variable_mapping)


# --- Regridding to CWA coordinate ---
era5_remap = xe.Regridder(era5, grid_cwa, method="bilinear")
era5_cwb = era5_remap(era5)


# --- Copy coordinate (latitude and longitude) ---
coord_list = ["XLAT", "XLAT_U", "XLAT_V", "XLONG", "XLONG_U", "XLONG_V"]
coords = {key: cwa.coords[key] for key in coord_list}
XTIME = np.datetime64("2024-11-26 15:00:00", "ns")
coords["XTIME"] = XTIME

cwb_channel = np.arange(4)
cwb_pressure = xr.DataArray(
    [np.nan, np.nan, np.nan, np.nan],
    dims=["cwb_channel"],
    coords={"cwb_channel": cwb_channel},
    name="cwb_pressure"
)

# Define variable names and create DataArray for cwb_variable
cwb_vnames = np.array(list(tccip.data_vars.keys()), dtype="<U26")
cwb_vars_dask = da.from_array(cwb_vnames, chunks=(4,))

# Now, define cwb_variable
cwb_variable = xr.DataArray(
    cwb_vars_dask,
    dims=["cwb_channel"],
    coords={
        "XTIME": ("cwb_channel", [XTIME] * len(cwb_channel)),
        "cwb_pressure": cwb_pressure
    },
    name="cwb_variable"
)


# define cwb
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


# Calculus cwb_center (mean)
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


# Calculus cwb_scale (std)
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


# define cwb_valid
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


#ERA5 write
era5_channel = np.arange(31)
plevs = [1000, 925, 850, 700, 500]  
era5_pressure_values = np.tile(plevs, 5) 
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


# define era5
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


# Calculus era5_center (mean)
era5_mean = da.stack(
    [
        era5.isel(era5_channel=channel).mean(dim=["time", "south_north", "west_east"]).data
        for channel in era5["era5_channel"].values
    ],
    axis=0
)

# define era5_center
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


# Calculus era5_scale (std)
era5_std = da.stack(
    [
        era5.isel(era5_channel=channel).std(dim=["time", "south_north", "west_east"]).data
        for channel in era5["era5_channel"].values
    ],
    axis=0
)

# define era5_scale
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


# define era5_valid
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


# output to the new file 
# Copy cwb coordinate and write new cwb and era5 data
# remove old XTIME and replace to new 
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


# write cwb data variables
out["cwb"] = cwb
out["cwb_center"] = cwb_center
out["cwb_scale"] = cwb_scale
out["cwb_valid"] = cwb_valid


out["era5"] = era5
out["era5_center"] = era5_center
out["era5_valid"] = era5_valid


out = out.drop_vars(["south_north", "west_east"])


# output zarr
comp = zarr.Blosc(cname='zstd', clevel=3, shuffle=2)
encoding = {var: {'compressor': comp} for var in out.data_vars}

outpath = 'corrdiff_testing_v241127.zarr'

with ProgressBar():
    out.to_zarr(outpath, mode='w', encoding=encoding, compute=True)

print(f"Data successfully saved to {outpath}")

