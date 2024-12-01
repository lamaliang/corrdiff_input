import os
import sys
import zarr
import xarray as xr
import numpy as np
from dask.diagnostics import ProgressBar

from tread import generate_tread_output
from era5 import generate_era5_output

##
# Functions
##

def generate_output_dataset(tread_file, era5_dir, grid, coords_cwa, start_date, end_date):
    # Generate CWB (i.e., TReAD) and ERA5 output fields.
    cwb, cwb_variable, cwb_center, cwb_scale, cwb_valid = \
        generate_tread_output(tread_file, grid, start_date, end_date)
    era5, era5_center, era5_scale, era5_valid = \
        generate_era5_output(era5_dir, grid, start_date, end_date)

    # Copy cwb coordinates and write new cwb and era5 coordinates; also replace XTIME with new coords.
    coords = {
        key: value.drop_vars("XTIME") if isinstance(value, (xr.DataArray, xr.Dataset)) and "XTIME" in value.coords else value
        for key, value in coords_cwa.items()
    }
    out = xr.Dataset(
        coords={
            "XLONG": coords["XLONG"],
            "XLAT": coords["XLAT"],
            "XLONG_U": coords["XLONG_U"],
            "XLAT_U": coords["XLAT_U"],
            "XLONG_V": coords["XLONG_V"],
            "XLAT_V": coords["XLAT_V"],
            "XTIME": np.datetime64("2024-11-26 15:00:00", "ns"), # TODO: check why
            "time": cwb.time  # 保留時間
        }
    )

    out = out.assign_coords(cwb_variable=cwb_variable)
    # TODO: check necessity
    # out = out.reset_coords("cwb_channel", drop=True) if "cwb_channel" in out.coords else out
    out.coords["era5_scale"] = ("era5_channel", era5_scale.data)

    # Write cwb & era5 data
    out["cwb"] = cwb
    out["cwb_center"] = cwb_center
    out["cwb_scale"] = cwb_scale
    out["cwb_valid"] = cwb_valid
    out["era5"] = era5
    out["era5_center"] = era5_center
    out["era5_valid"] = era5_valid

    out = out.drop_vars(["south_north", "west_east", "cwb_channel", "era5_channel"])
    print(out)

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
            "cwa_ref": "./data/cwa_dataset_example.zarr",
            "tread_file": f"./data/wrfo2D_d02_{yyyymm}.nc",
            "era5_dir": "./data",
        }

    # REMOTE
    return {
        "cwa_ref": "/lfs/home/dadm/data/cwa_dataset.zarr",
        "tread_file": f"/lfs/archive/TCCIP_data/TReAD/SFC/hr/wrfo2D_d02_{yyyymm}.nc",
        "era5_dir": "/lfs/archive/Reanalysis/ERA5",
    }

def generate_corrdiff_zarr(start_date, end_date):
    data_path = get_data_path(str(start_date)[:6])

    # Extract CorrDiff data's grid and coordinates for reference.
    cwa = xr.open_zarr(data_path["cwa_ref"])
    grid_cwa = xr.Dataset({ "lat": cwa.XLAT, "lon": cwa.XLONG })
    coords_cwa = {
            key: cwa.coords[key] for key in
            ["XLAT", "XLAT_U", "XLAT_V", "XLONG", "XLONG_U", "XLONG_V"]
        }

    out = generate_output_dataset( \
            data_path["tread_file"], data_path["era5_dir"], \
            grid_cwa, coords_cwa, \
            start_date, end_date)

    write_to_zarr(f"corrdiff_dataset_{start_date}_{end_date}.zarr", out)

def main():
    if len(sys.argv) < 3:
        print("Usage: python corrdiff_datagen.py <start_date> <end_date>")
        print("  e.g., $python corrdiff_datagen.py 20180101 20180103")
        sys.exit(1)

    generate_corrdiff_zarr(sys.argv[1], sys.argv[2])

if __name__ == "__main__":
    main()
