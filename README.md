# Run the code

## Set up environment via conda

```
conda env create -f corrdiff_input.yml
```

## Generate REF grid file for output zarr

```
cd ref_grid
python generate_wrf_coord.py
```

### Adjust REF grid size

1. Modify `ny` and `nx` in `generate_wrf_coord.py` to customize REF grid size:

```
ny, nx = 208, 208               # Desired grid dimensions
```

2. Revise `REF_GRID_NC` in `corrdiff_datagen.py` acccordingly:

```
REF_GRID_NC = "./ref_grid/wrf_208x208_grid_coords.nc"
```

## Prepare TReAD and ERA5 netcdf files (local testing only)
- Put TReAD file below under `data/tread/`.
  - `wrfo2D_d02_{yyyymm}.nc`
- Put ERA5 files below under `data/era5/`.
  - `ERA5_PRS_*_{yyyymm}_r1440x721_day.nc`
  - `ERA5_SFC_*_{yyyymm}_r1440x721_day.nc`
  - `ERA5_oro_r1440x721_day.nc`

### Example

```
data
├── era5
│   ├── ERA5_PRS_q_201801_r1440x721_day.nc
│   ├── ERA5_PRS_r_201801_r1440x721_day.nc
│   ├── ERA5_PRS_t_201801_r1440x721_day.nc
│   ├── ERA5_PRS_u_201801_r1440x721_day.nc
│   ├── ERA5_PRS_v_201801_r1440x721_day.nc
│   ├── ERA5_PRS_w_201801_r1440x721_day.nc
│   ├── ERA5_PRS_z_201801_r1440x721_day.nc
│   ├── ERA5_SFC_msl_201801_r1440x721_day.nc
│   ├── ERA5_SFC_t2m_201801_r1440x721_day.nc
│   ├── ERA5_SFC_tp_201801_r1440x721_day.nc
│   ├── ERA5_SFC_u10_201801_r1440x721_day.nc
│   ├── ERA5_SFC_v10_201801_r1440x721_day.nc
│   └── ERA5_oro_r1440x721.nc
└── tread
    └── wrfo2D_d02_201801.nc
```

## Generate CorrDiff dataset

```
python corrdiff_datagen.py <start_date> <end_date>
```
Example: `python corrdiff_datagen.py 20180101 20180105`

### Dump data for debugging
Set `DEBUG=True` in `corrdiff_datagen.py` to dump ERA5 & TReAD pre- and post-regridding data into netcdf files.

```
DEBUG = False  # Set to True to enable debugging
```

## Dump zarr file

```
python zarr_dump.py <zarr_file>
```
Example: `python zarr_dump.py corrdiff_dataset_20180101_20180105.zarr`

# Files

## Setup
- `corrdiff_input.yml`: conda enviroment `.yml` file.

## Generating REF grid
- `ref_grid/generate_wrf_coord.py`: Function to generate REF grid, with customized resolution and geographical area.
- `ref_grid/wrf_208x208_grid_coords.nc`: REF grid file for output zarr.
- `ref_grid/TReAD_wrf_d02_info.nc`: TReAD info file for REF grid generation.

## Generating dataset
- `corrdiff_datagen.py`: Main functions that take start & end dates, generate output dataset, and write to zarr file.
- `tread.py`: Functions to handle TReAD data.
- `era5.py`: Functions to handle ERA5 data.
- `util.py`: Helper functions.

## Debugging
- `zarr_dump.py`: Function to dump the generated zarr file.
