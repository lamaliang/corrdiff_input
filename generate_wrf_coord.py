import os
import numpy as np
from netCDF4 import Dataset

# === Parameters ===
clon, clat = 120.9465, 23.6745  # Center latitude and longitude
ny, nx = 208, 208               # Grid dimensions

# === Input file ===
infile = './data/TReAD_wrf_d02_info.nc'
nc_in = Dataset(infile, mode='r')

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
output_file = f"./data/wrf_{ny}x{nx}_grid_coords.nc"
if os.path.exists(output_file):
    os.remove(output_file)

with Dataset(output_file, mode="w", format="NETCDF4") as ncfile:
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

print(f"Output written to: {output_file}")
