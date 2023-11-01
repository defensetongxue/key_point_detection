python -u train.py --cfg ./configlist/hrnet_u.json
python -u test.py --cfg ./configlist/hrnet_u.json
python -u train.py --cfg ./configlist/hrnet_v.json
python -u test.py --cfg ./configlist/hrnet_v.json
python -u train.py --cfg ./configlist/hrnet.json 
python -u test.py --cfg ./configlist/hrnet.json 
python -u train.py --cfg ./configlist/hrnet_v1.json --save_name v1.pth
python -u test.py --cfg ./configlist/hrnet_v1.json --save_name v1.pth
python -u train.py --cfg ./configlist/hrnet_v2.json --save_name v2.pth
python -u test.py --cfg ./configlist/hrnet_v2.json --save_name v2.pth
python -u train.py --cfg ./configlist/hrnet_v3.json --save_name v3.pth
python -u test.py --cfg ./configlist/hrnet_v3.json --save_name v3.pth
python ring.py