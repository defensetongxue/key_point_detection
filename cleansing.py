from config import get_config
from utils_ import DRIONS_DB, GY_DB, HRF_DB, ODVOC_DB, STARE_DB
import os
# Parse arguments
args = get_config()

# Initialize dataset classes
datasets = {
    "DRIONS-DB": DRIONS_DB,
    "GY": GY_DB,
    "HRF": HRF_DB,
    "ODVOC": ODVOC_DB,
    "STARE": STARE_DB
}
os.makedirs(args.path_tar,exist_ok=True)

# Process each dataset
for dataset_name, dataset_class in datasets.items():
    print(f"Processing {dataset_name}...")
    source_path=os.path.join(args.path_src,dataset_name)
    target_path=os.path.join(args.path_tar,dataset_name)
    # remove exited data
    os.makedirs(target_path,exist_ok=True)
    os.system(f"rm -rf {target_path}/*")

    dataset = dataset_class(source_path, target_path)
    dataset.parse()
    print(f"Done processing {dataset_name}.")

print("All datasets processed successfully.")
