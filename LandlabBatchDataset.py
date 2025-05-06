import sqlite3
import os
import numpy as np

class LandlabBatchDataset:
    """
    A dataset for sets of landlab runs generated from a landlab batch run (github.com/jrymart/landlab-batcher).
    """

    def __init__(self, db_path, dataset_dir, label_query, filter_query="", trim=5):
        """
        Initialize the dataset with a path to the SQLite database.
        """
        self.db_path = db_path
        self.conn = sqlite3.connect(db_path)
        self.cursor = self.conn.cursor()
        self.dataset_dir = dataset_dir
        self.label_query = label_query
        self.filter_query = filter_query
        self.cursor.execute(f"SELECT model_run_id FROM runs {self.filter_query}")
        self.runs = [r[0] for r in self.cursor.fetchall()]
        self.trim = trim

    def __len__(self):
        return len(self.runs)

    def __getitem__(self, idx):
        run_name = self.runs[idx]
        data_path = os.path.join(self.dataset_dir, f"{run_name.npy}")
        self.cursor.execute(f"{self.label_query} WHERE model_run_id = \"{run_name}\"")
        label = self.cursor.fetchone()
        data_array = np.load(data_path)
        data_array = data_array.astype(np.float32)[self.trim:-self.trim, self.trim:-self.trim]
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
