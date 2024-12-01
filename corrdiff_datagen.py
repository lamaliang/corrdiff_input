import zarr
import xarray as xr
import numpy as np
from dask.diagnostics import ProgressBar

from tread import generate_tread_output
from era5 import generate_era5_output

##
# Configuration
##

iStart = 20180101
iLast = 20180103

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

def generate_output_dataset(tread_file, era5_dir, grid, coords_cwa, start_date, end_date):
    XTIME = np.datetime64("2024-11-26 15:00:00", "ns")
    coords = coords_cwa
    coords["XTIME"] = XTIME

    # Generate CWB (i.e., TReAD) and ERA5 output fields.
    cwb, cwb_variable, cwb_center, cwb_scale, cwb_valid = \
        generate_tread_output(tread_file, grid, start_date, end_date)
    era5, era5_center, era5_scale, era5_valid = \
        generate_era5_output(era5_dir, grid, start_date, end_date)

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

    out = out.drop_vars(["south_north", "west_east"])
    out = out.drop_vars(["cwb_channel", "era5_channel"])
    print(out)

    return out

def write_to_zarr(out_path, out_ds):
    comp = zarr.Blosc(cname='zstd', clevel=3, shuffle=2)
    encoding = { var: {'compressor': comp} for var in out_ds.data_vars }

    print('\n')
    with ProgressBar():
        out.to_zarr(out_path, mode='w', encoding=encoding, compute=True)

    print(f"Data successfully saved to [{out_path}]")


##
# Main
##

# Grid of CorrDiff data as reference
cwa = xr.open_zarr(data_path["cwa_ref"])
grid_cwa = xr.Dataset({ "lat": cwa.XLAT, "lon": cwa.XLONG })

# Copy coordinates "latitude" and "longitude"
coord_list = ["XLAT", "XLAT_U", "XLAT_V", "XLONG", "XLONG_U", "XLONG_V"]
coords_cwa = { key: cwa.coords[key] for key in coord_list }

out = generate_output_dataset( \
        data_path["tread_file"], data_path["era5_dir"], \
        grid_cwa, coords_cwa, \
        iStart, iLast)

write_to_zarr(f"corrdiff_dataset_{iStart}_{iLast}.zarr", out)