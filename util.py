import os
import xesmf as xe
import xarray as xr

def regrid_dataset(ds, grid) -> xr.Dataset:
    # Regrid the dataset to the target grid:
    # - Use bilinear interpolation to regrid the data.
    # - Extrapolate by using the nearest valid source cell to extrapolate values for target points outside the source grid.
    remap = xe.Regridder(ds, grid, method="bilinear", extrap_method="nearest_s2d")

    # Regrid each time step while keeping the original coordinates and dimensions
    ds_regrid = xr.concat(
        [remap(ds.isel(time=i)).assign_coords(time=ds.time[i])
            for i in range(ds.sizes["time"])],
        dim="time"
    )

    return ds_regrid

def verify_dataset(dataset):
    """
    Verifies an xarray.Dataset to ensure:
    1. Dimensions 'south_north' and 'west_east' are equal and both are multiples of 16.
    2. The dataset includes all specified coordinates and data variables.

    Parameters:
    - dataset: xarray.Dataset to verify.

    Returns:
    - A tuple (bool, str) where:
      - bool: True if the dataset passes all checks, False otherwise.
      - str: A message describing the result.
    """
    # Required dimensions, coordinates and data variables
    required_dims = [
        "time", "south_north", "west_east", "cwb_channel", "era5_channel"
    ]
    required_coords = [
        "time", "XLONG", "XLAT", "cwb_pressure", "cwb_variable",
        "era5_scale", "era5_pressure", "era5_variable"
    ]
    required_vars = [
        "cwb", "cwb_center", "cwb_scale", "cwb_valid",
        "era5", "era5_center", "era5_valid"
    ]

    # Check required dimensions
    missing_dims = [dim for dim in required_dims if dim not in dataset.dims]
    if missing_dims:
        return False, f"Missing required dimensions: {', '.join(missing_dims)}."
    if dataset.dims["south_north"] != dataset.dims["west_east"]:
        return False, "Dimensions 'south_north' and 'west_east' are not equal."
    if dataset.dims["south_north"] % 16 != 0:
        return False, "Dimensions 'south_north' and 'west_east' are not multiples of 16."

    # Check coordinates
    missing_coords = [coord for coord in required_coords if coord not in dataset.coords]
    if missing_coords:
        return False, f"Missing required coordinates: {', '.join(missing_coords)}."

    # Check data variables
    missing_vars = [var for var in required_vars if var not in dataset.data_vars]
    if missing_vars:
        return False, f"Missing required data variables: {', '.join(missing_vars)}."

    # All checks passed
    return True, "Dataset verification passed successfully."

def dump_regrid_netcdf(subdir, cwb_pre_regrid, cwb_post_regrid, era5_pre_regrid, era5_post_regrid) -> None:
    folder = f"./nc_dump/{subdir}/"
    os.makedirs(folder, exist_ok=True)

    cwb_pre_regrid.to_netcdf(folder + "tread_pre_regrid.nc")
    cwb_post_regrid.to_netcdf(folder + "tread_post_regrid.nc")
    era5_pre_regrid.to_netcdf(folder + "era5_pre_regrid.nc")
    era5_post_regrid.to_netcdf(folder + "era5_post_regrid.nc")

def is_local_testing() -> bool:
    return not os.path.exists("/lfs/archive/Reanalysis/")