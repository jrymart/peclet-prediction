from LandlabBatchDataset import build_datasets_from_db
import torch
import pandas as pd
import matplotlib.pyplot as plt

class PecletModelTrainer:
    """
    A class to train a model on Peclet number data.
    """

    def __init__(self, db_path, dataset_dir, model, label_query="SELECT peclet FROM model_run_outputs",
                 filter_query="", split_by="model_param.seed", train_fraction=.8, trim=5,
                 batch_size=64, epochs=5, learning_rate=0.001, **kwargs):
        """
        Initialize the trainer with a path to the SQLite database.
        """
        self.db_path = db_path
        self.dataset_dir = dataset_dir
        self.label_query = label_query
        self.filter_query = filter_query
        self.split_by = split_by
        self.train_fraction = train_fraction
        self.trim = trim
        self.epochs = epochs
        self.learning_rate = learning_rate
        self.model = model
        self.train_ds, self.test_ds = build_datasets_from_db(
            db_path,
            dataset_dir,
            label_query,
            filter_query,
            split_by,
            train_fraction,
            trim,
            **kwargs
        )
        self.train_loader = torch.utils.data.DataLoader(
            self.train_ds,
            batch_size=batch_size,
            shuffle=True
        )
        self.test_loader = torch.utils.data.DataLoader(
            self.test_ds,
            batch_size=batch_size,
            shuffle=False
        )

    def train(self, epochs=None, learning_rate=None, verbose=True):
        """"
        Train the model.
        """
        if epochs is None:
            epochs = self.epochs
        if learning_rate is None:
            learning_rate = self.learning_rate

        optimizer = torch.optim.Adam(self.model.parameters(), lr=learning_rate)
        criterion = torch.nn.MSELoss()

        for epoch in range(epochs):
            for i, data in enumerate(self.train_loader):
                data, labels, = data
                optimizer.zero_grad()
                outputs = self.model(data)
                loss = criterion(outputs, labels)
                loss.backward()
                optimizer.step()
            if verbose:
                print(f"Epoch {epoch+1}/{epochs}, Loss: {loss.item()}")

    def save_weights(self, path):
        """
        Save the model weights to a file.
        """
        torch.save(self.model.state_dict(), path)

    def load_weights(self, path):
        """
        Load the model weights from a file.
        """
        self.model.load_state_dict(torch.load(path))

    def evaluate(self):
        """
        Evaluate the model on the test set.
        """
        self.model.eval()
        total_loss = 0
        criterion = torch.nn.MSELoss()
        predictions = []
        true_labels = []
        #names = []
        with torch.no_grad():
            for i, data in enumerate(self.test_loader,0):
                data, labels = data
                outputs = self.model(data)
                loss = criterion(outputs, labels)
                total_loss += loss.item()
                predictions += outputs
                true_labels += labels
                #names += names.tolist()
        if self.test_loader.dataset.normalize:
            predictions = [p*self.test_loader.dataset.label_std + self.test_loader.dataset.label_mean for p in predictions]
            true_labels = [l*self.test_loader.dataset.label_std + self.test_loader.dataset.label_mean for l in true_labels]
        average_loss = total_loss / len(self.test_loader)
        self.test_df = pd.DataFrame({'predictions': [float(p) for p in predictions],
                                     'true_labels': [float(p) for p in true_labels]})
         #                            'names': names})
        print(f"Test Loss: {average_loss}")
