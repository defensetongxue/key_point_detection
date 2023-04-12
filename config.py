import argparse

def get_config():
    parser = argparse.ArgumentParser()
    # cleansing
    parser.add_argument('--path_src', type=str, default="../../autodl-tmp/datasets_original",
                        help='Path to the source folder containing original datasets.')
    parser.add_argument('--path_tar', type=str, default='../../autodl-tmp/datasets_keypoint',
                        help='Path to the target folder to store the processed datasets.')
    # Model
    parser.add_argument('--model', type=str, default='resnet18_regression',
                        help='Name of the model architecture to be used for training.')
    # train and test
    parser.add_argument('--dataset', type=str, default="GY",
                        help='Datset used. DRIONS-DB,GY,HRF,ODVOC,STARE | all')
    parser.add_argument('--save_name', type=str, default="best.pth",
                        help='Name of the file to save the best model during training.')
    parser.add_argument('--result_path', type=str, default="best.pth",
                        help='Path to the visualize result or the pytorch model will be saved.')
    parser.add_argument('--epoch', type=int, default=30,
                        help='Number of epochs for training.')
    parser.add_argument('--lr', type=float, default=1e-3,
                        help='Learning rate for the optimizer during training.')
    parser.add_argument('--batch_size', type=int, default=32,
                        help='batch size for dataloader in training and val.')
    parser.add_argument('--num_workers', type=int, default=4,
                        help='num_workers for dataloader in training and val.')
    parser.add_argument('--early_stop', type=int, default=10,
                        help='Number of epochs with no improvement to trigger early stopping.')
    parser.add_argument('--optimizer_type', type=str, default='Adam', help='Optimizer type (Adam or SGD)')

    args = parser.parse_args()
    return args
