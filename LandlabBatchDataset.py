import sqlite3
import os
import numpy as np

class LandlabBatchDataset:
    """
    A dataset for sets of landlab runs generated from a landlab batch run (github.com/jrymart/landlab-batcher).
    """

    def __init__(self, db_path, dataset_dir, label_query, filter_query="", trim=5, normalize=True, inputs_mean=None,
                 inputs_std=None, label_mean=None, label_std=None):
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
        cursor.execute(f"SELECT model_run_id FROM model_run_params")
        runs_for_normalization = [r[0] for r in self.cursor.fetchall()]
        if normalize:
            if inputs_mean is not None or inputs_std is not None:
                self.inputs_mean = inputs_mean
                self.inputs_std = inputs_std
            else:
                input_total_sum = 0.0
                input_total_sum_sq = 0.0
                input_total_count = 0
                for run_name in runs_for_normalization:
                    data_path = os.path.join(self.dataset_dir, f"{run_name}.npy")
                    data_array = np.load(data_path)
                    data_array = data_array.astype(np.float32)[self.trim:-self.trim, self.trim:-self.trim]
                    input_total_sum += np.sum(data_array)
                    input_total_sum_sq += np.sum(np.square(data_array))
                    input_total_count += data_array.size
                self.inputs_mean = input_total_sum / input_total_count
                variance = (input_total_sum_sq / input_total_count) - np.square(self.inputs_mean)
                if variance < 0 and np.isclose(variance, 0):
                    variance = 0.0
                self.inputs_std = np.sqrt(variance)
                if self.inputs_std <1e-7:
                    print("Warning: inputs_std is very small, setting to 1.0")
                    self.inputs_std = 1.0
            if label_mean is not None or label_std is not None:
                self.label_mean = label_mean
                self.label_std = label_std
            else:
                self.cursor.execute(f"{self.label_query} WHERE model_run_id IN ?", (runs_for_normalization,))
                labels = self.cursor.fetchall()
                all_labels = [l[0] for l in labels]
                all_labels_array = np.array(all_labels,dtype=np.float32)
                self.label_mean = np.mean(all_labels_array)
                self.label_std = np.std(all_labels_array)
                if self.label_std < 1e-7:
                    print("Warning: label_std is very small, setting to 1.0")
                    self.label_std = 1.0


    def __len__(self):
        return len(self.runs)

    def __getitem__(self, idx):
        run_name = self.runs[idx]
        data_path = os.path.join(self.dataset_dir, f"{run_name.npy}")
        self.cursor.execute(f"{self.label_query} WHERE model_run_id = \"{run_name}\"")
        label = self.cursor.fetchone()
        data_array = np.load(data_path)
        data_array = data_array.astype(np.float32)[self.trim:-self.trim, self.trim:-self.trim]
        data_array = (data_array - self.inputs_mean) / self.inputs_std
        label = (label - self.label_mean) / self.label_std
        return data_array, label

def build_datasets_from_db(db_path, dataset_dir, label_query, filter_query="", split_by="model_param.seed", train_fraction=.8, trim=5):
    """
    Create a LandlabBatchDataset instance.
    """
    conn = sqlite3.connect(db_path)
    cursor = conn.cursor()
    cursor.execute(f"SELECT DISTINCT \"{split_by}\" FROM runs {filter_query}")
    categories = [r[0] for r in cursor.fetchall()]
    split = int((len(categories) * train_fraction))
    train_categories = categories[:split]
    test_categories = categories[split:]
    train_filter = f"WHERE \"{split_by}\" IN ({', '.join([str(c) for c in train_categories])})"
    test_filter = f"WHERE \"{split_by}\" IN ({', '.join([str(c) for c in test_categories])})"
    train_ds = LandlabBatchDataset(db_path, dataset_dir, label_query, train_filter, trim)
    test_ds = LandlabBatchDataset(db_path, dataset_dir, label_query, test_filter, trim)
    return train_ds, test_ds
