import numpy as np

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