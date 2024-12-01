import unittest
from era5 import generate_era5_output, get_era5_center, get_era5_scale
import xarray as xr
import numpy as np

class TestERA5(unittest.TestCase):

    def setUp(self):
        # Mock input data
        cwa = xr.open_zarr("./data/cwa_dataset_example.zarr")
        self.grid = xr.Dataset({ "lat": cwa.XLAT, "lon": cwa.XLONG })

        # self.era5_out = xr.Dataset({
        #     "time": ("time", np.arange(5)),
        #     "south_north": ("south_north", np.arange(10)),
        #     "west_east": ("west_east", np.arange(10)),
        #     "temperature": (("time", "south_north", "west_east"), np.random.rand(5, 10, 10)),
        # })

        self.start_date = "20180101"
        self.end_date = "20180105"

    def test_generate_era5_output(self):
        # Mock ERA5 generation
        era5, era5_center, era5_scale, era5_valid = generate_era5_output(
            "./data", self.grid, self.start_date, self.end_date
        )
        self.assertIsInstance(era5, xr.DataArray)
        self.assertIsInstance(era5_center, xr.DataArray)
        self.assertIsInstance(era5_scale, xr.DataArray)
        self.assertIsInstance(era5_valid, xr.DataArray)

    # def test_get_era5_center(self):
    #     era5_center = get_era5_center(self.era5_out)
    #     self.assertEqual(era5_center.dims, ("era5_channel",))

    # def test_get_era5_scale(self):
    #     era5_scale = get_era5_scale(self.era5_out)
    #     self.assertEqual(era5_scale.dims, ("era5_channel",))
