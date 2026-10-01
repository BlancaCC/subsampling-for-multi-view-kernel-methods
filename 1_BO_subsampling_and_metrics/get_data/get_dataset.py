from get_data import dataset_info
from scipy.io import loadmat 
from numpy import array
dataset_root = '../Datasets'

def get_dataset(dataset_name, dataset_root=dataset_root):
    """Return the number of views, classes, and features per view for a dataset."""
    if dataset_name not in dataset_info:
        raise ValueError(f"Unknown dataset {dataset_name}.")
    n_views = dataset_info[dataset_name]['n_views']
    data = loadmat(f"{dataset_root}/{dataset_name}.mat")
    brute_X = data['X']
    y = data['y'].flatten()
    X_views = [
        array(brute_X[v][0]) for v in range(n_views)
    ]
    
    return X_views, y