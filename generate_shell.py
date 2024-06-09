import json,os
orignal_json_file='./config_file/class.json'
with open(orignal_json_file,'r') as f:
    orignal=json.load(f)
model_names=['vgg16','resnet18', 'resnet50' , 'mobilenetv3_large', 'mobilenetv3_small']
lr_lists=[5e-4,1e-4,5e-5,1e-5]
weight=[1.0,2.0,3.0,4.0]
cfg_cnt=0
sh_f=open('todo_cls.sh','w')
for m_n in model_names:
    for lr in lr_lists:
        for loss_w in weight:
            orignal["model"]["loss_weight"]=loss_w
            orignal["model"]["name"]=m_n
            orignal["train"]["lr"]=lr
            json_file_path=f'./cls_config/{str(cfg_cnt)}.json'
            with open(json_file_path,'w') as f:
                json.dump(orignal,f,indent=2)
                sh_f.write(f"python -u train_cls.py --cfg {json_file_path} --save_name {str(cfg_cnt)}_class.pth\n")
                sh_f.write(f"python -u test_cls.py --cfg {json_file_path} --save_name {str(cfg_cnt)}_class.pth\n")
            cfg_cnt+=1

sh_f.write("python -u ring.py\n")