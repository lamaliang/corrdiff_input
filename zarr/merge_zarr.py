import xarray as xr
import dask.array as da
import glob

# List and sort Zarr files
zarr_files = sorted(glob.glob('./corrdiff_*.zarr'))
for zarr_file in zarr_files:
    print(zarr_file)

# Open datasets with Dask
datasets = [xr.open_dataset(zarr_file, engine='zarr', chunks={}) for zarr_file in zarr_files]

# Concatenate datasets along the time dimension
combined = xr.concat(datasets, dim='time')

# Recompute era5_center and era5_scale
combined['era5_center'] = combined['era5'].mean(dim=['time', 'south_north', 'west_east'])
combined['era5_scale'] = combined['era5'].std(dim=['time', 'south_north', 'west_east'])

# Recompute cwb_center and cwb_scale
combined['cwb_center'] = combined['cwb'].mean(dim=['time', 'south_north', 'west_east'])
combined['cwb_scale'] = combined['cwb'].std(dim=['time', 'south_north', 'west_east'])

# Compute validity indicators
combined['cwb_valid'] = xr.DataArray(
    data=da.ones(len(combined["time"]), dtype=bool, chunks=len(combined["time"])),
    dims=["time"],
    coords={"time": combined["time"]},
    name="cwb_valid"
)

combined['era5_valid'] = xr.DataArray(
    data=da.ones((len(combined["time"]), len(combined["era5_channel"])), dtype=bool,
                 chunks=(len(combined["time"]), len(combined["era5_channel"]))),
    dims=["time", "era5_channel"],
    coords={
        "time": combined["time"],
        "era5_channel": combined["era5_channel"],
        "era5_pressure": combined["era5_pressure"],
        "era5_variable": combined["era5_variable"]
    },
    name="era5_valid"
)

combined = combined.drop_vars(["era5_channel"])
print(combined)

# Save the combined dataset to a new Zarr file
combined.to_zarr('./combined.zarr')
