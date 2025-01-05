import os
import sys
import zarr
import xarray as xr
import numpy as np
from dask.diagnostics import ProgressBar

from tread import generate_tread_output
from era5 import generate_era5_output
from util import dump_regrid_netcdf, print_slices_over_time

CORRDIFF_GRID_COORD_KEYS = ["XLAT", "XLONG"]

##
# Functions
##

def generate_output_dataset(tread_file, era5_dir, grid, grid_coords, start_date, end_date):
    # Generate CWB (i.e., TReAD) and ERA5 output fields.
    cwb, cwb_variable, cwb_center, cwb_scale, cwb_valid, cwb_pre_regrid, cwb_post_regrid = \
        generate_tread_output(tread_file, grid, start_date, end_date)
    era5, era5_center, era5_scale, era5_valid, era5_pre_regrid, era5_post_regrid = \
        generate_era5_output(era5_dir, grid, start_date, end_date)

    # Normalize both CWB and ERA5 time to 00:00:00, otherwise hour difference in-between causes data corruption after merging.
    cwb_normalized = cwb.assign_coords(time=cwb['time'].dt.floor('D'))
    era5_normalized = era5.assign_coords(time=era5['time'].dt.floor('D'))

    # Copy coordinates and remove XTIME if present
    coords = {
        key: value.drop_vars("XTIME") if "XTIME" in getattr(value, "coords", {}) else value
        for key, value in grid_coords.items()
    }

    # Create the output dataset
    out = xr.Dataset(
        coords={
            **{key: coords[key] for key in CORRDIFF_GRID_COORD_KEYS},
            "XTIME": np.datetime64("2024-12-24 20:00:00", "ns"),  # Placeholder for timestamp
            "time": cwb_normalized.time,
            "cwb_variable": cwb_variable,
            "era5_scale": ("era5_channel", era5_scale.data)
        }
    )

    # Assign CWB and ERA5 data variables
    out = out.assign({
        "cwb": cwb_normalized,
        "cwb_center": cwb_center,
        "cwb_scale": cwb_scale,
        "cwb_valid": cwb_valid,
        "era5": era5_normalized,
        "era5_center": era5_center,
        "era5_valid": era5_valid
    })

    out = out.drop_vars(["south_north", "west_east", "cwb_channel", "era5_channel"])

    # [DEBUG] Dump data pre- & post-regridding, and print output data slices.
    # dump_regrid_netcdf(cwb_pre_regrid, cwb_post_regrid, era5_pre_regrid, era5_post_regrid)
    # print_slices_over_time(out)

    return out

def write_to_zarr(out_path, out_ds):
    comp = zarr.Blosc(cname='zstd', clevel=3, shuffle=2)
    encoding = { var: {'compressor': comp} for var in out_ds.data_vars }

    print(f"\nSaving data to {out_path}:")
    with ProgressBar():
        out_ds.to_zarr(out_path, mode='w', encoding=encoding, compute=True)

    print(f"Data successfully saved to [{out_path}]")

def get_data_path(yyyymm):
    # LOCAL
    if not os.path.exists("/lfs/archive/Reanalysis/"):
        return {
            "coord_ref": "./data/wrf_r288x288_grid_coords.nc",
            "tread_file": f"./data/wrfo2D_d02_{yyyymm}.nc",
            "era5_dir": "./data/era5",
        }

    # REMOTE
    return {
        "coord_ref": "/lfs/home/lama/work/corrdiff_work/wrf_r288x288_grid_coords.nc",
        "tread_file": f"/lfs/archive/TCCIP_data/TReAD/SFC/hr/wrfo2D_d02_{yyyymm}.nc",
        "era5_dir": "/lfs/archive/Reanalysis/ERA5",
    }

def generate_corrdiff_zarr(start_date, end_date):
    data_path = get_data_path(str(start_date)[:6])

    # Extract REF grid.
    ref = xr.open_dataset(data_path["coord_ref"], engine='netcdf4')
    grid = xr.Dataset({ "lat": ref.XLAT, "lon": ref.XLONG })
    grid_coords = { key: ref.coords[key] for key in CORRDIFF_GRID_COORD_KEYS }

    out = generate_output_dataset( \
            data_path["tread_file"], data_path["era5_dir"], \
            grid, grid_coords, start_date, end_date)
    print(out)

    write_to_zarr(f"corrdiff_dataset_{start_date}_{end_date}.zarr", out)

def main():
    if len(sys.argv) < 3:
        print("Usage: python corrdiff_datagen.py <start_date> <end_date>")
        print("  e.g., $python corrdiff_datagen.py 20180101 20180103")
        sys.exit(1)

    generate_corrdiff_zarr(sys.argv[1], sys.argv[2])

if __name__ == "__main__":
    main()
