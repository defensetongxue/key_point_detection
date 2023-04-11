import numpy as np
def create_regression_label(annotation, image_width, image_height):
    keypoints = np.array(annotation['keypoints'], dtype=np.float32)
    keypoints[:, 0] /= image_width
    keypoints[:, 1] /= image_height

    return keypoints