import xarray as xr
import numpy as np
import os
import dask.array as da
import zarr
from dask.diagnostics import ProgressBar
import pandas as pd
import xesmf as xe

# === Date Setting ==
iStrt = 20180101
iLast = 20180102
yr = str(iStrt)[:4]
time = slice(str(iStrt),str(iLast))
yyyymm = str(iStrt)[:6]
plevs = [1000, 925, 850, 700, 500]

# === CWA Reference data ===
cwa = xr.open_zarr('/lfs/home/dadm/data/cwa_dataset.zarr')
lat_cwa = cwa.XLAT
lon_cwa = cwa.XLONG
grid_cwa = xr.Dataset({"lat": lat_cwa,"lon": lon_cwa,})

# ==================================================================

# === TCCIP TReAD data ===
ifnS = f"/lfs/archive/TCCIP_data/TReAD/SFC/hr/wrfo2D_d02_{yyyymm}.nc"
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

# --- Read Surface level data ---

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

# ==================================================================

# === ERA5 data ===
indir = '/lfs/archive/Reanalysis/ERA5'
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
era5_topo = xr.open_mfdataset("/lfs/archive/Reanalysis/ERA5/ERA5_oro_r1440x721.nc")[['oro']]
# -- units convert --- 
era5_sfc['tp'] = era5_sfc['tp'] * 24 * 1000
era5_sfc['tp'].attrs['units'] = 'mm/day'

# --- Merge prs,sfc,topo --- 
era5 = xr.merge([era5_prs,era5_sfc,era5_topo])

# --- Rename ---
variable_mapping = {
    "u": "eastward_wind",
    "v": "northward_wind",
    "t": "temperature",
    "r": "relative_humidity",
    "z": "geopotential_height",
    "msl": "surface_pressure",
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

print(tccip_cwb)
print(era5_cwb)

# ==================================================================
# === Covert to cwb_data.zarr format

# --- Copy coordinate (latitude and longitude) ---
coord_list = ["XLAT", "XLAT_U", "XLAT_V", "XLONG", "XLONG_U", "XLONG_V"]
coords = {key: cwa.coords[key] for key in coord_list}
coords["XTIME"] =  np.datetime64("2024-11-26 15:00:00")

cwb_channel = np.arange(4)
cwb_pressure = xr.DataArray(
    [np.nan, np.nan, np.nan, np.nan],
    dims=["cwb_channel"],
    coords={"cwb_channel": cwb_channel},
    name="cwb_pressure"
)

cwb_variable = xr.DataArray(
    list(tccip.data_vars.keys()), 
    dims=["cwb_channel"],
    coords={"cwb_channel": cwb_channel},
    name="cwb_variable"
)

print(cwb_variable)

#era5_channel = np.arange(31)
#pressure_values = np.tile(plevs, 5) 
#pressure_values = np.append(tccip_pressure_values, [np.nan] * 6)
#variables_values = [
#                    'eastward_wind','northward_wind','temperature','relative_humidity','geopotential_height',
#                    'eastward_wind','northward_wind','temperature','relative_humidity','geopotential_height',
#                    'eastward_wind','northward_wind','temperature','relative_humidity','geopotential_height',
#                    'eastward_wind','northward_wind','temperature','relative_humidity','geopotential_height',
#                    'eastward_wind','northward_wind','temperature','relative_humidity','geopotential_height',
#                    'mean_sea_level_pressure','precipitaion','temperature_2m','eastward_wind_10m','northward_wind_10m','terrain_height'
#                   ] 



#era5_pressure = xr.DataArray(
#    pressure_values,
#    dims=["era5_channel"],
#    coords={"era5_channel": era5_channel},
#    name="era5_pressure"
#)

#era5_variable = xr.DataArray(
#    variables_values,
#    dims=["era5_channel"],
#    coords={"era5_channel": era5_channel},
#    name="era5_variable"
#)

#coords["cwb_pressure"] = cwb_pressure
#coords["cwb_variable"] = cwb_variable
#coords["era5_pressure"]  = era5_pressure
#coords["era5_variable"]  = era5_variable


#out_ds = xr.Dataset(coords=coords)
#print(out_ds)
