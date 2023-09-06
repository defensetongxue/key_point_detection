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
    if cfg['train']['optimizer'] == 'sgd':
        optimizer = optim.SGD(
            filter(lambda p: p.requires_grad, model.parameters()),
            lr=cfg['train']['lr'],
            momentum=cfg['train']['momentum'],
            weight_decay=cfg['train']['wd'],
            nesterov=cfg['train']['nesterov']
        )
    elif cfg['train']['optimizer'] == 'adam':
        optimizer = optim.Adam(
            filter(lambda p: p.requires_grad, model.parameters()),
            lr=cfg['train']['lr']
        )
    elif cfg['train']['optimizer'] == 'rmsprop':
        optimizer = optim.RMSprop(
            filter(lambda p: p.requires_grad, model.parameters()),
            lr=cfg['train']['lr'],
            momentum=cfg['train']['momentum'],
            weight_decay=cfg['train']['wd'],
            alpha=cfg['train']['rmsprop_alpha'],
            centered=cfg['train']['rmsprop_centered']
        )
    else:
        raise
    return optimizer