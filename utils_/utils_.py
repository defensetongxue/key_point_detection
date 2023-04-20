import numpy as np
import inspect
from torch import optim

def contour_to_bbox(file_name):
    """
    Reads a text file containing contour coordinates and returns a bounding box in xywh format.
    :param file_name: str, the name of the text file
    :return: tuple containing the coordinates of the top-left corner and the dimensions of the bounding box (x, y, width, height)
    """
    with open(file_name, 'r') as f:
        lines = f.readlines()

    contour_points = [tuple(map(float, line.strip().split(','))) for line in lines]

    contour_array = np.array(contour_points)
    x_min, y_min = np.min(contour_array, axis=0)
    x_max, y_max = np.max(contour_array, axis=0)

    x_center, y_center = (x_min+x_max)/2, (y_min+y_max)/2
    width, height = x_max - x_min, y_max - y_min

    return x_center, y_center, width, height

def get_instance(module, class_name, *args, **kwargs):
    try:
        cls = getattr(module, class_name)
        instance = cls(*args, **kwargs)
        return instance
    except AttributeError:
        available_classes = [name for name, obj in inspect.getmembers(module, inspect.isclass) if obj.__module__ == module.__name__]
        raise ValueError(f"{class_name} not found in the given module. Available classes: {', '.join(available_classes)}")


def get_optimizer(cfg, model):
    optimizer = None
    if cfg.TRAIN.OPTIMIZER == 'sgd':
        optimizer = optim.SGD(
            filter(lambda p: p.requires_grad, model.parameters()),
            lr=cfg.TRAIN.LR,
            momentum=cfg.TRAIN.MOMENTUM,
            weight_decay=cfg.TRAIN.WD,
            nesterov=cfg.TRAIN.NESTEROV
        )
    elif cfg.TRAIN.OPTIMIZER == 'adam':
        optimizer = optim.Adam(
            filter(lambda p: p.requires_grad, model.parameters()),
            lr=cfg.TRAIN.LR
        )
    elif cfg.TRAIN.OPTIMIZER == 'rmsprop':
        optimizer = optim.RMSprop(
            filter(lambda p: p.requires_grad, model.parameters()),
            lr=cfg.TRAIN.LR,
            momentum=cfg.TRAIN.MOMENTUM,
            weight_decay=cfg.TRAIN.WD,
            alpha=cfg.TRAIN.RMSPROP_ALPHA,
            centered=cfg.TRAIN.RMSPROP_CENTERED
        )

    return optimizer