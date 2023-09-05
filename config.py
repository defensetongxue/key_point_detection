import argparse,json

def get_config():
    parser = argparse.ArgumentParser()
    # cleansing
    parser.add_argument('--path_src', type=str, default="../autodl-tmp/datasets_original",
                        help='Path to the source folder containing original datasets.')
    parser.add_argument('--path_tar', type=str, default='../autodl-tmp/datasets_keypoint',
                        help='Path to the target folder to store the processed datasets.')
    # Model
    parser.add_argument('--model', type=str, default='hrnet',
                        help='Name of the model architecture to be used for training.')
    
    # train and test
    parser.add_argument('--dataset', type=str, default="all",
                        help='Datset used. DRIONS-DB,GY,HRF,ODVOC,STARE | all')
    parser.add_argument('--save_name', type=str, default="./checkpoints/best.pth",
                        help='Name of the file to save the best model during training.')
    parser.add_argument('--result_path', type=str, default="experiments/visual",
                        help='Path to the visualize result or the pytorch model will be saved.')
    parser.add_argument('--from_checkpoint', type=str, default="",
                        help='load the exit checkpoint.')

    # config file 
    parser.add_argument('--cfg', help='experiment configuration filename',
                        default="./config/config_file/default.json", type=str)
    
    args = parser.parse_args()
    # Merge args and config file 
    args.configs=json.load(args.cfg)

    return args