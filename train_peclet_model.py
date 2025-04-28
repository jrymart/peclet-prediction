from LandlabBatchDataset import build_datasets_from_db
from ThreeLayerCNNRegressor import ThreeLayerCNNRegressor
import torch

class PecletModelTrainer:
    """
    A class to train a model on Peclet number data.
    """

    def __init__(self, db_path, dataset_dir, label_query="SELECT peclet FROM model_run_outputs",
                 filter_query="", split_by="model_param.seed", train_fraction=.8, trim=5,
                 batch_size=64, self.epeochs=5, learning_rate=0.001):
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
        self.model = ThreeLayerCNNRegressor()
        self.train_ds, self.test_ds = build_datasets_from_db(
            db_path,
            dataset_dir,
            label_query,
            filter_query,
            split_by,
            train_fraction,
            trim
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
            for data, labels, names in self.train_loader:
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
        names = []
        with torch.no_grad():
            for data, labels, names in self.test_loader:
                outputs = self.model(data)
                loss = criterion(outputs, labels)
                total_loss += loss.item()
                predictions += outputs.tolist()
                true_labels += labels.tolist()
                names += names.tolist()
        average_loss = total_loss / len(self.test_loader)
        print(f"Test Loss: {average_loss}")
