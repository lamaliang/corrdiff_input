"""
CorrDiff Dataset Generation and Zarr Storage.

This script processes TReAD and ERA5 datasets to generate a consolidated dataset,
verify its integrity, and save it in Zarr format. It integrates multiple data processing
modules and performs spatial regridding, variable aggregation, and data compression.

Features:
- Processes TReAD and ERA5 datasets for a specified date range.
- Regrids datasets to a reference grid.
- Generates a consolidated dataset with key variables and metrics:
  - Center (mean)
  - Scale (standard deviation)
  - Validity masks
- Verifies the structure and integrity of the dataset.
- Saves the dataset in compressed Zarr format.

Functions:
- `generate_output_dataset`: Combines processed TReAD and ERA5 data into a consolidated dataset.
- `write_to_zarr`: Writes the consolidated dataset to Zarr format with compression.
- `get_data_path`: Determines the paths for TReAD and ERA5 datasets based on the environment.
- `generate_corrdiff_zarr`: Orchestrates the generation, verification, and saving of the dataset.
- `main`: Parses command-line arguments and triggers the dataset generation process.

Dependencies:
- `sys`: For command-line argument parsing.
- `zarr`: For handling Zarr storage format.
- `xarray`: For multi-dimensional labeled data operations.
- `numpy`: For numerical operations.
- `dask.diagnostics.ProgressBar`: For monitoring progress during dataset writing.
- Modules:
  - `tread`: For TReAD dataset processing.
  - `era5`: For ERA5 dataset processing.
  - `util`: For utility functions like dataset verification and regridding.

Usage:
    python corrdiff_datagen.py <start_date> <end_date>

    Example:
        python corrdiff_datagen.py 20180101 20180103

Notes:
- Ensure that the `REF_GRID_NC` file exists and contains valid reference grid data.
- The script handles both local and remote environments based on the presence of specific folders.

"""
import sys
import zarr
import xarray as xr
import numpy as np
from dask.diagnostics import ProgressBar

from tread import generate_tread_output
from era5 import generate_era5_output
from util import is_local_testing, verify_dataset, dump_regrid_netcdf

CORRDIFF_GRID_COORD_KEYS = ["XLAT", "XLONG"]
REF_GRID_NC = "./ref_grid/wrf_208x208_grid_coords.nc"

##
# Functions
##

def generate_output_dataset(tread_file, era5_dir, grid, grid_coords, start_date, end_date):
    """
    Generates a consolidated output dataset by processing TReAD and ERA5 data fields.

    Parameters:
        tread_file (str): Path to the TReAD file or directory containing the TReAD dataset.
        era5_dir (str): Path to the directory containing ERA5 datasets.
        grid (xr.Dataset): Reference grid dataset for spatial alignment.
        grid_coords (dict): Dictionary of grid coordinate arrays.
        start_date (str): Start date of the data range in 'YYYYMMDD' format.
        end_date (str): End date of the data range in 'YYYYMMDD' format.

    Returns:
        xr.Dataset: A dataset containing consolidated and processed TReAD and ERA5 data fields.
    """
    # Generate CWB (i.e., TReAD) and ERA5 output fields.
    cwb, cwb_variable, cwb_center, cwb_scale, cwb_valid, cwb_pre_regrid, cwb_post_regrid = \
        generate_tread_output(tread_file, grid, start_date, end_date)
    era5, era5_center, era5_scale, era5_valid, era5_pre_regrid, era5_post_regrid = \
        generate_era5_output(era5_dir, grid, start_date, end_date)

    # Copy coordinates and remove XTIME if present
    coords = {
        key: value.drop_vars("XTIME") if "XTIME" in getattr(value, "coords", {}) else value
        for key, value in grid_coords.items()
    }

    # Create the output dataset
    out = xr.Dataset(
        coords={
            **{key: coords[key] for key in CORRDIFF_GRID_COORD_KEYS},
            "XTIME": np.datetime64("2025-01-07 17:00:00", "ns"),  # Placeholder for timestamp
            "time": cwb.time,
            "cwb_variable": cwb_variable,
            "era5_scale": ("era5_channel", era5_scale.data)
        }
    )

    # Assign CWB and ERA5 data variables
    out = out.assign({
        "cwb": cwb,
        "cwb_center": cwb_center,
        "cwb_scale": cwb_scale,
        "cwb_valid": cwb_valid,
        "era5": era5,
        "era5_center": era5_center,
        "era5_valid": era5_valid
    })

    out = out.drop_vars(["south_north", "west_east", "cwb_channel", "era5_channel"])

    # [DEBUG] Dump data pre- & post-regridding, and print output data slices.
    # dump_regrid_netcdf(f"{start_date}_{end_date}", \
    #     cwb_pre_regrid, cwb_post_regrid, era5_pre_regrid, era5_post_regrid)

    return out

def write_to_zarr(out_path, out_ds):
    """
    Writes the given dataset to a Zarr storage format with compression.

    Parameters:
        out_path (str): The file path where the Zarr dataset will be saved.
        out_ds (xr.Dataset): The dataset to be written to Zarr format.

    Returns:
        None
    """
    comp = zarr.Blosc(cname='zstd', clevel=3, shuffle=2)
    encoding = { var: {'compressor': comp} for var in out_ds.data_vars }

    print(f"\nSaving data to {out_path}:")
    with ProgressBar():
        out_ds.to_zarr(out_path, mode='w', encoding=encoding, compute=True)

    print(f"Data successfully saved to [{out_path}]")

def get_data_path():
    """
    Determines the data paths for TReAD and ERA5 datasets based on the environment.

    Returns:
        dict: A dictionary containing paths to TReAD and ERA5 data directories.
    """
    # LOCAL
    if is_local_testing():
        return {
            "tread_dir": "./data/tread",
            "era5_dir": "./data/era5",
        }

    # REMOTE
    return {
        "tread_dir": "/lfs/archive/TCCIP_data/TReAD/SFC/hr",
        "era5_dir": "/lfs/archive/Reanalysis/ERA5",
    }

def generate_corrdiff_zarr(start_date, end_date):
    """
    Generates and verifies a consolidated dataset for TReAD and ERA5 data,
    then writes it to a Zarr file format.

    Parameters:
        start_date (str): Start date of the data range in 'YYYYMMDD' format.
        end_date (str): End date of the data range in 'YYYYMMDD' format.

    Returns:
        None
    """
    data_path = get_data_path()

    # Extract REF grid.
    ref = xr.open_dataset(REF_GRID_NC, engine='netcdf4')
    grid = xr.Dataset({ "lat": ref.XLAT, "lon": ref.XLONG })
    grid_coords = { key: ref.coords[key] for key in CORRDIFF_GRID_COORD_KEYS }

    # Generate the output dataset.
    out = generate_output_dataset( \
            data_path["tread_dir"], data_path["era5_dir"], \
            grid, grid_coords, start_date, end_date)
    print(f"\nZARR dataset =>\n {out}")

    # Verify the output dataset.
    passed, message = verify_dataset(out)
    if not passed:
        print(f"\nDataset verification failed => {message}")
        return

    # Write the output dataset to ZARR.
    write_to_zarr(f"corrdiff_dataset_{start_date}_{end_date}.zarr", out)

def main():
    """
    Main entry point for the script. Parses command-line arguments to generate
    a Zarr dataset for a specified date range.

    Command-line Usage:
        python corrdiff_datagen.py <start_date> <end_date>

    Example:
        python corrdiff_datagen.py 20180101 20180103

    Returns:
        None
    """
    if len(sys.argv) < 3:
        print("Usage: python corrdiff_datagen.py <start_date> <end_date>")
        print("  e.g., $python corrdiff_datagen.py 20180101 20180103")
        sys.exit(1)

    generate_corrdiff_zarr(sys.argv[1], sys.argv[2])

if __name__ == "__main__":
    main()
