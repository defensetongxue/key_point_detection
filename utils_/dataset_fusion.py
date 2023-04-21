import os 
from Datasets_ import CustomDatset
def get_dataset(path_tar,data_name):
    train_dataset,val_dataset=None,None
    datasets_list=["DRIONS-DB","GY","HRF","ODVOC"]
    if data_name=='all':
        for dataset in datasets_list:
            data_path=os.path.join(path_tar, dataset)
            if train_dataset:
                train_dataset += CustomDatset(data_path,split="train")
                val_dataset += CustomDatset(data_path,split="valid")
            else:
                train_dataset = CustomDatset(data_path,split="train")
                val_dataset = CustomDatset(data_path,split="valid")
    else:
        data_path=os.path.join(path_tar, data_name)
        train_dataset = CustomDatset(data_path,split="train")
        val_dataset = CustomDatset(data_path,split="valid")
    return train_dataset,val_dataset