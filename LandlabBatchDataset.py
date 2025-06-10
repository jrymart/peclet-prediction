import sqlite3
import os
import numpy as np
from torch.utils.data import Dataset
from torchvision.transforms import ToTensor, Lambda
import torch

class LandlabBatchDataset(Dataset):
    """
    A dataset for sets of landlab runs generated from a landlab batch run (github.com/jrymart/landlab-batcher).
    """

    def __init__(self, db_path, dataset_dir, label_query, filter_query="", trim=5, normalize=True, inputs_mean=None,
                 inputs_std=None, labels_mean=None, labels_std=None):
        """
        Initialize the dataset with a path to the SQLite database.
        """
        self.db_path = db_path
        self.conn = sqlite3.connect(db_path)
        self.cursor = self.conn.cursor()
        self.dataset_dir = dataset_dir
        self.label_query = label_query
        self.filter_query = filter_query
        self.cursor.execute(f"SELECT model_run_id FROM model_run_params {self.filter_query}")
        self.runs = [r[0] for r in self.cursor.fetchall()]
        self.trim = trim
        self.normalize = normalize
        self.cursor.execute(f"SELECT model_run_id FROM model_run_params")
        runs_for_normalization = [r[0] for r in self.cursor.fetchall()]
        self.normalize = normalize
        self.inputs_mean = inputs_mean
        self.inputs_std = inputs_std
        self.labels_mean = labels_mean
        self.labels_std = labels_std
        self.transform = ToTensor()

    def __len__(self):
        return len(self.runs)

    def __getitem__(self, idx):
        run_name = self.runs[idx]
        data_path = os.path.join(self.dataset_dir, f"{run_name}.npy")
        self.cursor.execute(f"{self.label_query} WHERE model_run_id = \"{run_name}\"")
        label = self.cursor.fetchone()[0]
        data_array = np.load(data_path)
        if data_array.ndim == 2:
            data_array = data_array.astype(np.float32)[self.trim:-self.trim, self.trim:-self.trim]
        else:
            data_array = data_array.astype(np.float32)[:, self.trim:-self.trim, self.trim:-self.trim]
        if self.normalize:
            data_array = (data_array - self.inputs_mean) / self.inputs_std
            label =(label - self.labels_mean) / self.labels_std
        return self.transform(data_array), torch.tensor(label, dtype=torch.float32)

def get_dataset_stats(db_path, dataset_dir, filter_query="", trim=5):
    """
    Get the mean and std of the dataset.
    """
    conn = sqlite3.connect(db_path)
    cursor = conn.cursor()
    cursor.execute(f"SELECT model_run_id FROM model_run_params {filter_query}")
    runs = [r[0] for r in cursor.fetchall()]
    data_array = np.load(os.path.join(dataset_dir, f"{runs[0]}.npy"))
    if data_array.ndim ==3:
        sum = np.zeros(data_array.shape[0], dtype=np.float32)
        sum_sq = np.zeros(data_array.shape[0], dtype=np.float32)
        count = np.zeros(data_array.shape[0], dtype=np.int64)
    else:
        sum = 0.0
        sum_sq = 0.0
        count = 0
    for run_name in runs:
        data_path = os.path.join(dataset_dir, f"{run_name}.npy")
        data_array = np.load(data_path)
        if data_array.ndim ==3:
            data_array = data_array.astype(np.float32)[:, trim:-trim, trim:-trim]
            sum += np.sum(data_array, axis=(1, 2))
            sum_sq += np.sum(np.square(data_array), axis=(1, 2))
            count += data_array.shape[1] * data_array.shape[2]
        else:
            data_array = data_array.astype(np.float32)[trim:-trim, trim:-trim]
            sum += np.sum(data_array)
            sum_sq += np.sum(np.square(data_array))
            count += data_array.size
    mean = sum / count
    variance = (sum_sq / count) - np.square(mean)
    handle_variance = np.any([variance < 0, np.isclose(variance, 0)])
    if np.array(variance).ndim == 1:
        if handle_variance:
            variance = 0
    else:
        variance[handle_variance] = 0
    std = np.sqrt(variance)
    handle_std = std < 1e-7
    if np.array(std).ndim == 1:
        if handle_std:
            std = 1
        print("Warning: std is very small, setting to 1")
    else:
        std[handle_std] = 1
        print("Warning: std is very small, setting to 1")
    if data_array.ndim == 3:
        mean = mean[:, np.newaxis, np.newaxis]
        std = std[:, np.newaxis, np.newaxis]
    return mean, std

def get_label_stats(db_path, label_query, filter_query=""):
    """
    Get the mean and std of the labels.
    """
    conn = sqlite3.connect(db_path)
    cursor = conn.cursor()
    cursor.execute(f"SELECT model_run_id FROM model_run_params {filter_query}")
    runs = [r[0] for r in cursor.fetchall()]
    labels = []
    limit = conn.getlimit(sqlite3.SQLITE_LIMIT_VARIABLE_NUMBER)
    for i in range(0, len(runs), limit):
        current_chunk_runs = runs[i:i + limit]
        placeholders = ', '.join(['?'] * len(current_chunk_runs))
        cursor.execute(f"{label_query} WHERE model_run_id IN ({placeholders})", current_chunk_runs)
        current_labels = [l[0] for l in cursor.fetchall()]
        labels += current_labels
    labels = np.array(labels)
    mean = np.mean(labels)
    std= np.std(labels)
    if std < 1e-7:
        print("Warning: std is very small, setting to 1")
        std = 1
    return mean, std

def build_datasets_from_db(db_path, dataset_dir, label_query, filter_query="", split_by="model_param.seed",
                           train_fraction=.8, trim=5, normalize=True, inputs_mean=None,inputs_std=None,
                           labels_mean=None, labels_std=None, **kwargs):
    """
    Create a LandlabBatchDataset instance.
    """
    conn = sqlite3.connect(db_path)
    cursor = conn.cursor()
    cursor.execute(f"SELECT DISTINCT \"{split_by}\" FROM model_run_params {filter_query}")
    categories = [r[0] for r in cursor.fetchall()]
    split = int((len(categories) * train_fraction))
    train_categories = categories[:split]
    test_categories = categories[split:]
    train_filter = f"WHERE \"{split_by}\" IN ({', '.join([str(c) for c in train_categories])})"
    test_filter = f"WHERE \"{split_by}\" IN ({', '.join([str(c) for c in test_categories])})"
    if normalize:
        if labels_mean is None or labels_std is None:
            labels_mean, labels_std = get_label_stats(db_path, label_query, train_filter)
        if inputs_mean is None or inputs_std is None:
            inputs_mean, inputs_std = get_dataset_stats(db_path, dataset_dir, train_filter, trim)
    train_ds = LandlabBatchDataset(db_path, dataset_dir, label_query, train_filter, trim, normalize, inputs_mean,
                                   inputs_std, labels_mean, labels_std)
    test_ds = LandlabBatchDataset(db_path, dataset_dir, label_query, test_filter, trim, normalize, inputs_mean,
                                   inputs_std, labels_mean, labels_std)
    return train_ds, test_ds
