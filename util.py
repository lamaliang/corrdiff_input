import os
import xesmf as xe
import xarray as xr

def regrid_dataset(ds, grid):
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

def dump_regrid_netcdf(cwb_pre_regrid, cwb_post_regrid, era5_pre_regrid, era5_post_regrid):
    folder = "./nc_dump/"
    os.makedirs(folder, exist_ok=True)

    cwb_pre_regrid.to_netcdf(folder + "tread_pre_regrid.nc")
    cwb_post_regrid.to_netcdf(folder + "tread_post_regrid.nc")
    era5_pre_regrid.to_netcdf(folder + "era5_pre_regrid.nc")
    era5_post_regrid.to_netcdf(folder + "era5_post_regrid.nc")

def print_slices_over_time(ds, limit=10):
    print("\n" + "-"*40)

    end = min(ds.time.size, limit)
    for t in ds.time[:end]:
        # Select the data for the current time step
        cwb_data = ds['cwb'].sel(time=t)
        era5_data = ds['era5'].sel(time=t)
        half_shape_x = 288 // 2

        print(f"\nTime: {t.values} Half_Shape_X: {half_shape_x}")
        print("CWB Data Slice:")
        print(cwb_data[0, half_shape_x].values[half_shape_x - 10: half_shape_x + 10])
        print("ERA5 Data Slice:")
        print(era5_data[0, half_shape_x].values[half_shape_x - 10: half_shape_x + 10])

    print("\n" + "-"*40 + "\n")