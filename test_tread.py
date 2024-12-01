import unittest
from tread import generate_tread_output, get_cwb_center, get_cwb_scale
import xarray as xr
import numpy as np

class TestTREAD(unittest.TestCase):

    def setUp(self):
        # Mock input data
        cwa = xr.open_zarr("./data/cwa_dataset_example.zarr")
        self.grid = xr.Dataset({ "lat": cwa.XLAT, "lon": cwa.XLONG })

        # self.tread_out = xr.Dataset({
        #     "time": ("time", np.arange(5)),
        #     "south_north": ("south_north", np.arange(10)),
        #     "west_east": ("west_east", np.arange(10)),
        #     "precipitation": (("time", "south_north", "west_east"), np.random.rand(5, 10, 10)),
        # })

        self.start_date = "20180101"
        self.end_date = "20180105"

    def test_generate_tread_output(self):
        cwb, cwb_variable, cwb_center, cwb_scale, cwb_valid = generate_tread_output(
            "./data/wrfo2D_d02_201801.nc", self.grid, self.start_date, self.end_date
        )
        self.assertIsInstance(cwb, xr.DataArray)
        self.assertIsInstance(cwb_center, xr.DataArray)
        self.assertIsInstance(cwb_scale, xr.DataArray)
        self.assertIsInstance(cwb_valid, xr.DataArray)

    # def test_get_cwb_center(self):
    #     cwb_center = get_cwb_center(self.tread_out, None, None)
    #     self.assertEqual(cwb_center.dims, ("cwb_channel",))

    # def test_get_cwb_scale(self):
    #     cwb_scale = get_cwb_scale(self.tread_out, None, None)
    #     self.assertEqual(cwb_scale.dims, ("cwb_channel",))
