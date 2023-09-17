from sklearn.metrics import accuracy_score, f1_score, roc_auc_score, confusion_matrix
from sklearn.preprocessing import label_binarize
import numpy as np

def get_criteria(preds, distance, position_gt, distance_gt):
    # Classification Metrics
    distance_gt_one_hot = label_binarize(distance_gt, classes=[0, 1, 2])
    distance_one_hot = label_binarize(distance, classes=[0, 1, 2])

    acc = accuracy_score(distance_gt, distance)
    auc = roc_auc_score(distance_gt_one_hot, distance_one_hot, multi_class='ovo')
    f1 = f1_score(distance_gt, distance, average='weighted')

    # Location Metrics
    mse = np.mean((preds - position_gt)**2)
    
    # Creating a criteria dictionary to store all metric values
    criteria_dict = {
        "Accuracy": acc,
        "AUC": auc,
        "F1 Score": f1,
        "MSE": mse
    }
    
    return criteria_dict
