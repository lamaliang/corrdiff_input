import zarr
import sys
import xarray as xr

from tread import TREAD_CHANNELS
from util import print_slices_over_time

def dump_zarr_fields(zarr_path):
    """
    Dumps specified fields from a Zarr file and prints their content and structure.

    Parameters:
        zarr_path (str): Path to the Zarr file.
    """
    # Open the Zarr file in consolidated mode
    group = zarr.open_consolidated(zarr_path)

    print("Zarr Group Tree Structure:")
    print(group.tree())
    print("\nField Details:\n")

    # Iterate over the specified folders and dump the content
    for field_group, count in [
        ("cwb_center", len(TREAD_CHANNELS)),
        ("cwb_pressure", len(TREAD_CHANNELS)),
        ("cwb_scale", len(TREAD_CHANNELS)),
        ("cwb_variable", len(TREAD_CHANNELS)),
        ("era5_center", 21),
        ("era5_pressure", 21),
        ("era5_scale", 21),
        ("era5_variable", 21)
    ]:
        print(f"\n{field_group}:")
        print(group[field_group].info)
        for i in range(count):
            print(f"  Index {i}: {group[field_group][i]}")

    # Print general information about all groups
    # print("\nAll Groups Information:\n")
    # for folder in group:
    #     print(f"{folder}:")
    #     print(group[folder].info)

    print("\nZarr Dataset Structure:")
    ds = xr.open_zarr(zarr_path)
    print(ds)

    print_slices_over_time(ds)

def main():
    """
    Main function to parse positional argument for Zarr file path.
    """
    if len(sys.argv) < 2:
        print("Usage: python script.py <file_path>")
        sys.exit(1)

    zarr_path = sys.argv[1]  # First positional argument
    dump_zarr_fields(zarr_path)

if __name__ == "__main__":
    main()

