import xesmf as xe
import xarray as xr

def regrid_dataset(ds, grid, grid_dims):
    ds_regrid = xr.Dataset(coords={
        "time": ds["time"],  # 保留原始的時間座標
        "XLAT": grid["XLAT"],
        "XLONG": grid["XLONG"]
    })

    # 重新網格化每個時間步，並保留時間座標
    remap = xe.Regridder(ds, grid, method="bilinear")
    for var in ds.data_vars:
        if set(ds[var].dims) >= grid_dims:
            # 遍歷時間維度
            regridded_list = [
                remap(ds[var].isel(time=i))
                .assign_coords(time=ds["time"].isel(time=i))  # 重新指派原始時間座標
                for i in range(ds.sizes["time"])
            ]
            # 合併所有時間切片，並沿時間維度堆疊
            ds_regrid[var] = xr.concat(regridded_list, dim="time")

    return ds_regrid