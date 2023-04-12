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

if not os.path.exists(args.path_tar):
    os.mkdir(args.path_tar)

# Process each dataset
for dataset_name, dataset_class in datasets.items():
    print(f"Processing {dataset_name}...")
    source_path=os.path.join(args.path_src,dataset_name)
    target_path=os.path.join(args.path_tar,dataset_name)
    if not os.path.exists(target_path):
        os.mkdir(target_path)
    dataset = dataset_class(source_path, target_path)
    dataset.process()
    print(f"Done processing {dataset_name}.")

print("All datasets processed successfully.")
