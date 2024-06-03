import numpy as np
import os
from glob import glob
try:
    import torch
except:
    pass
import logging

def set_logging(logdir, logname="run.log"):
    new_logname = os.path.join(logdir, logname)

    # Remove all handlers associated with the root logger
    root_logger = logging.getLogger()
    for handler in root_logger.handlers[:]:
        root_logger.removeHandler(handler)

    # Set up new handlers
    logging.basicConfig(filename=new_logname,
                        filemode='a',
                        format='%(asctime)s %(levelname)s %(message)s',
                        datefmt='%Y-%m-%d %H:%M:%S',
                        level=logging.INFO)
    
    console_handler = logging.StreamHandler()
    console_handler.setLevel(logging.INFO)
    console_formatter = logging.Formatter('%(asctime)s %(levelname)s %(message)s', datefmt='%Y-%m-%d %H:%M:%S')
    console_handler.setFormatter(console_formatter)
    root_logger.addHandler(console_handler)


class StandardScaler:
    """A standard scaler for data normalization."""

    def __init__(self, mean, std):
        self.mean = mean
        self.std = std

    def transform(self, data):
        """Normalize the data."""
        return (data - self.mean) / self.std

    def inverse_transform(self, data):
        """Revert the normalization."""
        return (data * self.std) + self.mean


def load_data_from_files(data_path, levels):
    """
    Load and sort data file paths based on levels. Ex: [train_1, train_2], [test_1, test_2]

    Args:
        data_path (str): Path to the data directory.
        levels (int): Number of fidelities.

    Returns:
        train_paths, test_paths: Sorted list of file paths.
    """
    train_paths = []
    test_paths = []
    for level in range(levels):
        train_paths.append((os.path.join(data_path, f"train_l{level + 1}.npz")))
        test_paths.append((os.path.join(data_path, f"test_l{level + 1}.npz")))
    return train_paths, test_paths

def split_data(x, y, train_ratio=0.9):
    train_size = int(train_ratio * len(x))
    train_x, valid_x = x[:train_size], x[train_size:]
    train_y, valid_y = y[:train_size], y[train_size:]
    return train_x, valid_x, train_y, valid_y

def get_dataset(data_path, levels, transform=True):
    """
    Create a dataset from the specified path.

    Args:
        data_path (str): Path to the data directory.
        levels (int): Number of fidelities.
        transform (bool): Whether to apply standard scaling.

    Returns:
        dict: A dictionary containing the dataset.
    """
    data = {"train_x": [], "train_y": [], "valid_x": [], "valid_y": [], "test_x": [], "test_y": [],
            "scaler_x": None, "scaler_y": [], "levels": levels}

    output_dims = []
    train_paths, test_paths = load_data_from_files(data_path, levels)
    
    for train_path, test_path in zip(train_paths, test_paths):
        train_data = np.load(train_path)
        test_data = np.load(test_path)

        x, y = train_data['x'], train_data['y']
        train_x, valid_x, train_y, valid_y = split_data(x, y)
        test_x, test_y = test_data['x'], test_data['y']

        if transform:
            if "train_l1" in train_path: # initalize scaler
                scaler_x = StandardScaler(np.mean(train_x), np.std(train_x))
                data["scaler_x"] = scaler_x
            else:
                scaler_x = data["scaler_x"]

            scaler_y = StandardScaler(np.mean(train_y), np.std(train_y))
            data[f"scaler_y"].append(scaler_y)

            train_x = scaler_x.transform(train_x)
            train_y = scaler_y.transform(train_y)
            valid_x = scaler_x.transform(valid_x)
            valid_y = scaler_y.transform(valid_y)
            test_x = scaler_x.transform(test_x)
            test_y = scaler_y.transform(test_y)

        data["train_x"].append(train_x)
        data["train_y"].append(train_y)
        data["valid_x"].append(valid_x)
        data["valid_y"].append(valid_y)
        data["test_x"].append(test_x)
        data["test_y"].append(test_y)
        output_dims.append(train_y.shape[1])
    
    input_dim = train_x.shape[1]
    return input_dim, output_dims, data


class MultiFidelityDataLoader:
    """A DataLoader for multi-fidelity datasets."""
    def __init__(self, data, device, batch_size, valid=False, test=False):
        self.data = data
        self.device = device

        self.train = not valid and not test
        self.name = "train"
        if valid:
            self.name = "valid"
        elif test:
            self.name = "test"

        max_batch_size = min([level_data.shape[0] for level_data in self.data[f"{self.name}_x"]])
        self.batch_size = max_batch_size

        self.len_iter = max_batch_size * (self.data[f"{self.name}_x"][0].shape[0] // max_batch_size)
        self.len = max(1, self.len_iter // max_batch_size)

    def __iter__(self):
        """Yield mini-batches of data."""
        for b in range(0, self.len_iter, self.batch_size):
            xs = [torch.from_numpy(level_data[b:b + self.batch_size].astype(np.float32)).to(self.device)
                  for level_data in self.data[f"{self.name}_x"]]
            ys = [torch.from_numpy(level_data[b:b + self.batch_size].astype(np.float32)).to(self.device)
                  for level_data in self.data[f"{self.name}_y"]]
            yield xs, ys

    def shuffle(self):
        """Shuffle the training data."""
        if self.train:
            for level in range(self.data["levels"]):
                idx = np.random.permutation(len(self.data[f"train_x"][level]))
                for key in ("x", "y"):
                    self.data[f"train_{key}"][level] = self.data[f"train_{key}"][level][idx]


class TFMultiFidelityDataLoader:
    """A DataLoader for multi-fidelity datasets."""
    def __init__(self, data, device, batch_size, valid=False, test=False):
        self.data = data
        self.device = device

        self.train = not valid and not test
        self.name = "train"
        if valid:
            self.name = "valid"
        elif test:
            self.name = "test"

        max_batch_size = min([level_data.shape[0] for level_data in self.data[f"{self.name}_x"]])
        self.batch_size = max_batch_size

        self.len_iter = max_batch_size * (self.data[f"{self.name}_x"][0].shape[0] // max_batch_size)
        self.len = max(1, self.len_iter // max_batch_size)

    def __iter__(self):
        """Yield mini-batches of data."""
        for b in range(0, self.len_iter, self.batch_size):
            xs = [level_data[b:b + self.batch_size].astype(np.float32).to(self.device)
                  for level_data in self.data[f"{self.name}_x"]]
            ys = [level_data[b:b + self.batch_size].astype(np.float32).to(self.device)
                  for level_data in self.data[f"{self.name}_y"]]
            yield xs, ys

    def shuffle(self):
        """Shuffle the training data."""
        if self.train:
            for level in range(self.data["levels"]):
                idx = np.random.permutation(len(self.data[f"train_x"][level]))
                for key in ("x", "y"):
                    self.data[f"train_{key}"][level] = self.data[f"train_{key}"][level][idx]
