"""
NetCDF Grid Extraction and Resampling.

This script extracts a subgrid or resamples the input NetCDF file to a new
resolution centered at a specified latitude and longitude.

Features:
- **Grid Extraction**: Extracts a `(ny, nx)` subset centered on (`clon`, `clat`).
- **Grid Resampling**: Uses bilinear interpolation (xESMF) if upscaling is needed.
- **Metadata Preservation**: Copies essential variables
  (`XLAT`, `XLONG`, `TER`, `LANDMASK`, `SLOPE`, `ASPECT`).
- **Automatic File Handling**: Deletes existing output before saving.

### Parameters:
- `clon`, `clat` (float): Center longitude & latitude for extraction.
- `ny`, `nx` (int): Target grid size.
- `INPUT_FILE` (str): Path to the input NetCDF file.
- `OUTPUT_FILE` (str): Path for the extracted/resampled output.

### Workflow:
1. **Load Input Data**: Reads `XLAT`, `XLONG`, terrain (`TER`), and land properties.
2. **Determine Extraction Mode**:
   - **Grid Smaller Than Input** → Crop around center.
   - **Grid Larger Than Input** → Use xESMF to resample.
3. **Save Extracted/Resampled Data**: Writes to `OUTPUT_FILE` in NETCDF4 format.

### Dependencies:
- `numpy`, `netCDF4`, `xesmf`, `xarray`, `pathlib`.

### Example Usage:
1. Update `clon`, `clat`, `ny`, `nx`, and `INPUT_FILE`.
2. Run the script to generate `OUTPUT_FILE`.

### Notes:
- **Ensure `INPUT_FILE` contains required variables**.
- **Supports both cropping and upscaling** via interpolation.
- **Output file is automatically replaced if it exists**.
"""
from pathlib import Path
import numpy as np
from netCDF4 import Dataset
import xesmf as xe
import xarray as xr

# === Parameters ===
clon, clat = 120.9465, 23.6745  # Center latitude and longitude
ny, nx = 208, 208               # Desired grid dimensions

# === Input file ===
INPUT_FILE = './TReAD_wrf_d02_info.nc'
OUTPUT_FILE = f"./wrf_{ny}x{nx}_grid_coords.nc"

# Load input dataset, including:
# - latitude & longitude
# - terrain height & landmask
# - orography slope & aspect
nc_in = Dataset(INPUT_FILE, mode='r')
lat, lon = nc_in.variables['XLAT'][:], nc_in.variables['XLONG'][:]
ter, lmask = nc_in.variables['TER'][:], nc_in.variables['LANDMASK'][:]
slope, aspect = nc_in.variables['SLOPE'][:], nc_in.variables['ASPECT'][:]

# Check if extrapolation is needed
print(f"Input grid (lat, lon) = ({lat.shape[0]}, {lon.shape[1]})")
if ny > lat.shape[0] or nx > lon.shape[1]:
    print("Extrapolating to larger grid...")

    # Create new lat/lon grid
    new_lat = np.linspace(lat.min(), lat.max(), ny)
    new_lon = np.linspace(lon.min(), lon.max(), nx)
    new_grid = xr.Dataset({
        "lat": (["south_north", "west_east"], np.meshgrid(new_lat, new_lon, indexing='ij')[0]),
        "lon": (["south_north", "west_east"], np.meshgrid(new_lat, new_lon, indexing='ij')[1])
    })

    # Use xESMF for extrapolation
    regridder = xe.Regridder(
        xr.Dataset({"lat": (["south_north", "west_east"], lat),
                    "lon": (["south_north", "west_east"], lon)}),
        new_grid, method="bilinear", extrap_method="nearest_s2d"
    )

    lat_grid, lon_grid = new_grid["lat"].values, new_grid["lon"].values
    ter, lmask, slope, aspect = [regridder(xr.DataArray(var))
                                 for var in (ter, lmask, slope, aspect)]
else:
    print("Cropping to smaller grid...")

    # Find center indices
    idy, idx = np.abs(lat[:, 0] - clat).argmin(), np.abs(lon[0, :] - clon).argmin()

    # Calculate slicing indices
    slat, elat = max(0, idy - ny // 2), min(lat.shape[0], idy + ny // 2)
    slon, elon = max(0, idx - nx // 2), min(lon.shape[1], idx + nx // 2)

    # Crop the grid
    lat_grid, lon_grid = lat[slat:elat, slon:elon], lon[slat:elat, slon:elon]
    ter, lmask, slope, aspect = [var[slat:elat, slon:elon]
                                 for var in (ter, lmask, slope, aspect)]

# === Save to Output File ===
output_path = Path(OUTPUT_FILE)
if output_path.exists():
    output_path.unlink(missing_ok=True)

with Dataset(OUTPUT_FILE, mode="w", format="NETCDF4") as ncfile:
    # Create dimensions
    ncfile.createDimension("south_north", lat_grid.shape[0])
    ncfile.createDimension("west_east", lon_grid.shape[1])

    for name, data, unit in [("XLAT", lat_grid, "degrees_north"),
                             ("XLONG", lon_grid, "degrees_east"),
                             ("TER", ter, "meters"),
                             ("LANDMASK", lmask, "land mask"),
                             ("SLOPE", slope, "slope"),
                             ("ASPECT", aspect, "degree")]:
        var = ncfile.createVariable(name, "f4", ("south_north", "west_east"))
        var[:, :] = data
        var.units = unit

    ncfile.setncattr("coordinates", "XLAT XLONG")
    ncfile.setncattr("description", f"CorrDiff REF grid {ny}x{nx}")

print(f"Output written to => {OUTPUT_FILE}")
