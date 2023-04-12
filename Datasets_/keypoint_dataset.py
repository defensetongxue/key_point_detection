from .dataset_regression import KeypointDetectionDatasetRegression
from .dataset_heatmap import KeypointDetectionDatasetHeatmap

def get_keypoint_dataset(path, split, output_format, **kwargs):
    if output_format == 'regression':
        return KeypointDetectionDatasetRegression(path, split, **kwargs)
    elif output_format == 'heatmap':
        return KeypointDetectionDatasetHeatmap(path, split, **kwargs)
    else:
        raise ValueError(f"Invalid output_format: {output_format}")