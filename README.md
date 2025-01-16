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

## Prepare TReAD and ERA5 `.nc` files (local testing only)
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

## Dump zarr file

```
python zarr_dump.py <zarr_file>
```
Example: `python zarr_dump.py corrdiff_dataset_20180101_20180105.zarr`

## Dump data for debugging
Uncomment [lines](https://github.com/bentian/corrdiff_input/blob/cfc60d0a32a8c208bbf100cfd0f838b204fb4077/corrdiff_datagen.py#L59) in `corrdiff_datagen.py`.

# Files

## Setup
- `corrdiff_input.yml`: conda enviroment `.yml` file.

## Generating REF grid
- `ref_grid/generate_wrf_coord.py`: Function to generate REF grid, with customized resolution and geographical area.
- `ref_grid/wrf_208x208_grid_coords.nc`: REF grid file for output zarr.
- `ref_grid/TReAD_wrf_d02_info.nc`: TReAD info file for REF grid generation.

## Generating dataset
- `corrdiff_datagen.py`: Main functions that take start & end dates, generate output dataset, and write to zarr file.
- `tread.py`: Functions that handle TReAD data.
- `era5.py`: Functions that handle ERA5 data.
- `util.py`: Helper functions.

## Debugging
- `zarr_dump.py`: Function to dump the generated zarr file.
