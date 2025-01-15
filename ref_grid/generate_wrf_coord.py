"""
Grid Extraction and NetCDF Output Generation.

This script extracts a grid subset centered around a specified latitude and longitude
from an input NetCDF file. The extracted grid data is then saved into a new NetCDF file
with appropriate metadata.

Features:
- Extracts a grid of specified dimensions (`ny` x `nx`) centered on given coordinates.
- Handles bounds to ensure slicing remains within the data range.
- Copies relevant variables (`XLAT`, `XLONG`, `TER`, `LANDMASK`) from the input file.
- Saves the extracted grid into a new NetCDF file with metadata and attributes.

Parameters:
- clon, clat (float): Center longitude and latitude of the grid.
- ny, nx (int): Dimensions of the grid to extract.
- INPUT_FILE (str): Path to the input NetCDF file containing the source data.
- OUTPUT_FILE (str): Path where the extracted grid will be saved.

Workflow:
1. Open the input NetCDF file and read the required variables.
2. Locate the center point based on the specified coordinates.
3. Compute slicing indices for the desired grid dimensions.
4. Extract the grid and relevant variables.
5. Create a new NetCDF file and save the extracted data with metadata.

Dependencies:
- `os`: For file handling.
- `numpy`: For numerical operations.
- `netCDF4`: For working with NetCDF files.

Example Usage:
    1. Update the parameters `clon`, `clat`, `ny`, `nx`, and `INPUT_FILE`.
    2. Run the script to generate an output file containing the grid.

Notes:
- Ensure the input file (`INPUT_FILE`) exists and contains
  the required variables: `XLAT`, `XLONG`, `TER`, and `LANDMASK`.
- The script automatically removes any existing output file at the specified `OUTPUT_FILE` path.
"""
import os
import numpy as np
from netCDF4 import Dataset

# === Parameters ===
clon, clat = 120.9465, 23.6745  # Center latitude and longitude
ny, nx = 208, 208               # Grid dimensions

# === Input file ===
INPUT_FILE = './TReAD_wrf_d02_info.nc'
nc_in = Dataset(INPUT_FILE, mode='r')

lat = nc_in.variables['XLAT'][:]
lon = nc_in.variables['XLONG'][:]
ter = nc_in.variables['TER'][:]
lmask = nc_in.variables['LANDMASK'][:]

# Find index of the center point
idy = np.abs(lat[:, 0] - clat).argmin()
idx = np.abs(lon[0, :] - clon).argmin()

# Calculate the slicing indices
slat_idx = max(0, idy - ny // 2)
elat_idx = min(lat.shape[0], slat_idx + ny)
slon_idx = max(0, idx - nx // 2)
elon_idx = min(lat.shape[1], slon_idx + nx)

# Extract the grid
lat_grid = lat[slat_idx:elat_idx, slon_idx:elon_idx].astype("float32")
lon_grid = lon[slat_idx:elat_idx, slon_idx:elon_idx].astype("float32")
ter_grid = ter[slat_idx:elat_idx, slon_idx:elon_idx].astype("float32")
lmask_grid = lmask[slat_idx:elat_idx, slon_idx:elon_idx].astype("float32")

# === Output file ===
OUTPUT_FILE = f"./wrf_{ny}x{nx}_grid_coords.nc"
if os.path.exists(OUTPUT_FILE):
    os.remove(OUTPUT_FILE)

with Dataset(OUTPUT_FILE, mode="w", format="NETCDF4") as ncfile:
    # Create dimensions
    ncfile.createDimension("south_north", lat_grid.shape[0])
    ncfile.createDimension("west_east", lon_grid.shape[1])

    # Create variables
    nlat   = ncfile.createVariable("XLAT", "f4", ("south_north", "west_east"))
    nlon   = ncfile.createVariable("XLONG", "f4", ("south_north", "west_east"))
    nter   = ncfile.createVariable("TER", "f4", ("south_north", "west_east"))
    nlmask = ncfile.createVariable("LANDMASK", "f4", ("south_north", "west_east"))

    # Assign values
    nlat[:,:]   = lat_grid
    nlon[:,:]   = lon_grid
    nter[:,:]   = ter[slat_idx:elat_idx, slon_idx:elon_idx]
    nlmask[:,:] = lmask[slat_idx:elat_idx, slon_idx:elon_idx]

    # Add coordinate metadata
    ncfile.setncattr("coordinates", "XLAT XLONG")

    # Add attributes
    nlat.units = "degrees_north"
    nlon.units = "degrees_east"
    nter.units = "meters"
    nlmask.units = "land mask"
    ncfile.setncattr("description", f"New CorrDiff Training REF grid {ny}x{nx}")

print(f"Output written to => {OUTPUT_FILE}")
