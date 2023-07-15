import zarr
import os

# zarr_path = "data/block_pushing/multimodal_push_seed.zarr"
zarr_path = "data/table/table_test.zarr"
zarr_file = zarr.open(zarr_path, mode="r")

group = zarr.open(os.path.expanduser(zarr_path), "r")
print(group.tree())