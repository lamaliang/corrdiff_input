import unittest
import xarray as xr
import numpy as np
from corrdiff_datagen import generate_output_dataset, CORRDIFF_GRID_COORD_KEYS

class TestCorrDiffDatagen(unittest.TestCase):

    def setUp(self):
        # Mock input data
        cwa = xr.open_zarr("./data/cwa_dataset_example.zarr")
        self.grid = xr.Dataset({ "lat": cwa.XLAT, "lon": cwa.XLONG })
        self.grid_coords = { key: cwa.coords[key] for key in CORRDIFF_GRID_COORD_KEYS }

        # self.grid = xr.Dataset({
        #     "lat": (["south_north", "west_east"], np.random.rand(10, 10)),
        #     "lon": (["south_north", "west_east"], np.random.rand(10, 10))
        # })
        # self.grid_coords = {
        #     "XLAT": xr.DataArray(np.random.rand(10, 10), dims=["south_north", "west_east"]),
        #     "XLONG": xr.DataArray(np.random.rand(10, 10), dims=["south_north", "west_east"]),
        # }

        self.tread_file = "./data/wrfo2D_d02_201801.nc"
        self.era5_dir = "./data/era5"
        self.start_date = "20180101"
        self.end_date = "20180103"

    def test_generate_output_dataset(self):
        # Mock dataset generation
        output = generate_output_dataset(
            self.tread_file,
            self.era5_dir,
            self.grid,
            self.grid_coords,
            self.start_date,
            self.end_date
        )

        # Validate output dataset structure
        self.assertIsInstance(output, xr.Dataset)
        self.assertIn("cwb", output.data_vars)
        self.assertIn("era5", output.data_vars)
        self.assertIn("cwb_center", output.data_vars)
        self.assertIn("era5_scale", output.coords)
