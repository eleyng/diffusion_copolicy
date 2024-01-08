'''
This script prints the length of the dataset in zarr format.
'''

import zarr
import os

zarr_path = "data/table/table_10Hz.zarr"
zarr_file = zarr.open(zarr_path, mode="r")

group = zarr.open(os.path.expanduser(zarr_path), "r")
print(group.tree())