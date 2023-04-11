import numpy as np

def create_heatmap_label(annotation, image_width, image_height, output_width, output_height, sigma=2):
    num_keypoints = len(annotation['keypoints'])
    heatmap = np.zeros((num_keypoints, output_height, output_width), dtype=np.float32)

    for i, (x, y) in enumerate(annotation['keypoints']):
        x = int(x * output_width / image_width)
        y = int(y * output_height / image_height)
        
        # Create a Gaussian heatmap around the keypoint location
        xx, yy = np.meshgrid(np.arange(output_width), np.arange(output_height), sparse=True)
        heatmap[i] = np.exp(-((xx - x) ** 2 + (yy - y) ** 2) / (2 * sigma ** 2))

    return heatmap