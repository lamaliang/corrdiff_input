"""
Module for filtering a Zarr dataset based on extreme weather event dates.

This module extracts and merges extreme event dates from two text files—one containing
typhoon days and another listing extreme precipitation events—and filters a Zarr dataset
to retain only the specified dates.

### **Main Features**
- Extracts dates from a file containing date **ranges**.
- Extracts dates from a file containing **individual dates**.
- Merges, sorts, and filters dates to keep only **extreme event occurrences**.
- Filters an **xarray Zarr dataset** to include only these extreme dates.
- Saves the **filtered dataset** to a new Zarr file.

### **Key Functions**
1. `extract_overlapping_dates(file_path: str) -> set`
   - Extracts and returns unique dates from a file containing **date ranges**.

2. `read_dates_from_file(file_path: str) -> set`
   - Extracts and returns unique dates from a file containing **individual date entries**.

3. `filter_zarr_by_dates(file1: str, file2: str, zarr_path: str, output_path: str) -> None`
   - Filters a Zarr dataset by merging extreme dates from two sources and keeping only those dates.

4. `main()`
   - Defines file paths and calls `filter_zarr_by_dates()` to generate a filtered dataset.
"""
import re
from datetime import datetime, timedelta
import xarray as xr
import numpy as np

from merge_zarr import recompute_fields

def extract_overlapping_dates(file_path):
    """
    Reads a file containing date ranges and extracts unique overlapping dates.

    Parameters:
        file_path (str): Path to the input file.

    Returns:
        List[str]: Sorted list of unique dates in YYYYMMDD format.
    """
    unique_dates = set()

    with open(file_path, "r", encoding="utf-8") as file:
        for line in file:
            match = re.match(r"(\d{10})-(\d{10})", line.strip())
            if match:
                start_date = datetime.strptime(match.group(1)[:8], "%Y%m%d")
                end_date = datetime.strptime(match.group(2)[:8], "%Y%m%d")

                # Generate all dates from start to end (inclusive)
                while start_date <= end_date:
                    unique_dates.add(start_date.strftime("%Y%m%d"))
                    start_date += timedelta(days=1)

    return unique_dates

def read_dates_from_file(file_path):
    """
    Reads a file containing a list of individual dates.

    Parameters:
        file_path (str): Path to the input file.

    Returns:
        set: Unique set of dates in YYYYMMDD format.
    """
    with open(file_path, "r", encoding="utf-8") as file:
        return {line.strip() for line in file if re.match(r"\d{8}", line.strip())}

def filter_zarr_by_dates(file1: str, file2: str, zarr_path: str, output_path: str) -> None:
    """
    Filters the Zarr dataset to include only the specified extreme dates and
    saves the filtered dataset.

    Parameters:
        zarr_path (str): Path to the input Zarr dataset.
        file1 (str): Path to the first input file containing date ranges.
        file2 (str): Path to the second input file containing individual dates.
        output_path (str): Path to save the filtered dataset.
    """
    # Extract dates from both files
    dates_from_file1 = extract_overlapping_dates(file1)
    dates_from_file2 = read_dates_from_file(file2)

    # Merge, sort, and remove duplicates
    extreme_dates = sorted(dates_from_file1.union(dates_from_file2))

    # Save to output file
    # with open("../data/extreme_dates/extreme_dates.txt", "w") as file:
    #     for date in extreme_dates:
    #         file.write(date + "\n")

    # Open the Zarr dataset
    ds = xr.open_zarr(zarr_path, consolidated=True)

    # Ensure the time coordinate is in datetime64 format
    if not np.issubdtype(ds.time.dtype, np.datetime64):
        ds["time"] = ds.indexes["time"].to_datetimeindex()  # Convert to datetime

    # Convert dataset time to YYYYMMDD format
    available_dates = np.array(
        [str(np.datetime_as_string(t, unit="D")).replace("-", "")
         for t in ds.time.values],dtype=str
    )

    # Filter to keep only extreme dates
    selected_times = ds.time[np.isin(available_dates, list(extreme_dates))]
    filtered_ds = recompute_fields(ds.sel(time=selected_times))
    print(filtered_ds)

    # Save filtered dataset
    filtered_ds.to_zarr(output_path, mode="w")

    print(f"\nFiltered dataset saved to [{output_path}]")

def main():
    """
    Filters a Zarr dataset to include only extreme weather event dates.

    Workflow:
        1. **Define File Paths:**
           - `input_zarr`: The full dataset containing all available dates.
           - `typhoon_dates_file`: A text file listing typhoon days (JTWC dataset).
           - `extreme_prcp_file`: A text file listing extreme precipitation days.
           - `output_zarr`: The destination for the filtered dataset.

        2. **Filter Dataset by Extreme Dates:**
           - Calls `filter_zarr_by_dates()` to extract only the dates in the extreme event lists.

        3. **Save Filtered Data:**
           - The filtered dataset is saved as `extreme_dataset.zarr`.

    Inputs:
        - `../archive/merged_dataset_00-23.zarr`: The full input dataset.
        - `../data/extreme_dates/TCdays_300km_JTWC_2000_2023.txt`: List of typhoon-related dates.
        - `../data/extreme_dates/2000_2023_MJ_extremed1.txt`: List of extreme precipitation dates.

    Outputs:
        - `"extreme_dataset.zarr"`: A dataset containing only the selected extreme weather dates.

    Notes:
        - The function assumes the **input Zarr file** has a **"time" coordinate**.
        - The text files must contain **dates in YYYYMMDD format**, one per line.
        - The filtering function `filter_zarr_by_dates()` should handle merging and sorting.
    """
    input_zarr = "../archive/merged_dataset_00-23.zarr"
    typhoon_dates_file = "../data/extreme_dates/TCdays_300km_JTWC_2000_2023.txt"
    extreme_prcp_file = "../data/extreme_dates/2000_2023_MJ_extremed1.txt"
    output_zarr = "extreme_dataset.zarr"

    filter_zarr_by_dates(typhoon_dates_file, extreme_prcp_file, input_zarr, output_zarr)

if __name__ == "__main__":
    main()
