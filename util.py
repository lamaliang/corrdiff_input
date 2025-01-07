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

def dump_regrid_netcdf(cwb_pre_regrid, cwb_post_regrid, era5_pre_regrid, era5_post_regrid) -> None:
    folder = "./nc_dump/"
    os.makedirs(folder, exist_ok=True)

    cwb_pre_regrid.to_netcdf(folder + "tread_pre_regrid.nc")
    cwb_post_regrid.to_netcdf(folder + "tread_post_regrid.nc")
    era5_pre_regrid.to_netcdf(folder + "era5_pre_regrid.nc")
    era5_post_regrid.to_netcdf(folder + "era5_post_regrid.nc")

def is_local_testing() -> bool:
    return not os.path.exists("/lfs/archive/Reanalysis/")