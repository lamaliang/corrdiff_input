import xesmf as xe
import xarray as xr

def regrid_dataset(ds, grid):
    # Regrid data over the spatial dimensions for all timestamps, based on CWA coordinates.

    remap = xe.Regridder(ds, grid, method="bilinear",extrap_method=None)
    ds_regrid = xr.concat(
        [remap(ds.isel(time=i)).assign_coords(time=ds.time[i])
            for i in range(ds.sizes["time"])],
        dim="time"
    )

    return ds_regrid
