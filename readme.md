
# Optic Disc Location  
[官中版](./说明.md)

This repository contains the official code for the optic disc location module, mainly responsible for training three models: optic disc location under visible conditions, optic disc location under non-visible conditions, and the distance of the optic disc from the field of view under non-visible conditions.

For optic disc location, this repository uses a heatmap approach, which generates a $H*W$ heatmap, where $H$ and $W$ are parameters related to the original image size, here set to 0.25 times the original size. By analyzing the heatmap, the most probable region for the optic disc is identified. For generating heatmaps of optic discs outside the field of view, the approach involves using the furthest point with a larger sigma (wider range for the label) and slightly attenuated value. Refer to `cleansing.py` or the supplementary materials of the original paper for the code.

If you have any questions, feel free to raise them in the issues section or email me. 

## Main File Descriptions
If the optic disc location is annotated, the `annotations.json` field will generate an `optic_disc_gt`. Example:

```python
{
    "1.jpg":{
        "image_path":"data_path/images/1.jpg"
        "id":1,
        "optic_disc_gt":{
            'position':[x,y],
            "distance": "near" | "far"
        }
    }
    ...
}
```
- `cleansing.py`: Generates the corresponding optic disc heatmap based on optic disc annotations.
- `exclude_unvisiual.py`: Divides the original data into `u_xxx.json` and `v_xxx.json`, handling visible and non-visible optic disc cases separately. I have tried mixed processing, such as including some non-visible optic disc data in visible cases, but found no improvement in results. Thus, I recommend handling these cases separately.
- `config.py`: Stores configuration files. The configuration is divided into two parts: model and some training-related parameters (generally not adjusted) in the config_file's JSON file, and the rest in this file.
- `train.py`: Trains the localization model.
- `train_cls.py`: Trains the classification model for the distance of non-visible optic discs to the screen.
- `test.py`: Tests the localization model. This file is used to test the localization model for visible optic discs by outputting its distribution among different classes (visible or non-visible) to find the appropriate threshold.
- `test_cls.py`: Tests the classification model for the distance of non-visible optic discs to the screen.
