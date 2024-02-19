# python train_cls.py --split_name 1 --cfg ./config_file/class.json --save_name optic_disc_cls.pth
# python train_cls.py --split_name 2 --cfg ./config_file/class.json --save_name optic_disc_cls.pth
# python train_cls.py --split_name 3 --cfg ./config_file/class.json --save_name optic_disc_cls.pth
# python train_cls.py --split_name u_4 --cfg ./config_file/class.json --save_name optic_disc_cls.pth
# python generate_optic_disc.py --cfg ./config_file/defalut.json --split_name 1
python generate_optic_disc.py --cfg ./config_file/defalut.json --split_name 2
python generate_optic_disc.py --cfg ./config_file/defalut.json --split_name 3
python generate_optic_disc.py --cfg ./config_file/defalut.json --split_name 4