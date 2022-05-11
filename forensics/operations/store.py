import os
import os.path
import pandas as pd
import logging
from sklearn.preprocessing import StandardScaler
from dataclasses import dataclass

_cRESULT = "class"
_cID = "txId"
_cTIME = "timestep"
_fFEATURES_LAKE = "data/lake/elliptic_txs_features.csv"
_fLABELS_LAKE = "data/lake/elliptic_txs_classes.csv"
_fSTORE = {
    "features": "data/store/store_features_processed.csv",
    "labels": "data/store/store_labels_processed.csv",
}


@dataclass
class DataLake:
    features: pd.DataFrame
    labels: pd.DataFrame
    # edges: pd.DataFrame


@dataclass
class TrainingData:
    features: pd.DataFrame
    labels: pd.DataFrame


class EllipticDataStore:
    training_data: TrainingData

    def __init__(self, force_rebuild=False, **kwargs):
        if len(kwargs) < 2:
            kwargs = _fSTORE
        working_dir = os.getcwd()
        features_store = os.path.join(working_dir, kwargs["features"])
        labels_store = os.path.join(working_dir, kwargs["labels"])
        if os.path.exists(features_store) and force_rebuild == False:
            features_df = pd.read_csv(features_store)
            labels_df = pd.read_csv(labels_store)
            self.training_data = TrainingData(features_df, labels_df)
        else:
            processor = EllipticDataProcessor()
            logging.info("Request dataset with force_rebuild=%s" % force_rebuild)
            data = processor.get_full_dataset(force_rebuild)
            logging.info("Store processed features: " + os.path.abspath(features_store))
            data.features.to_csv(features_store)
            logging.info("Store processed labels: " + os.path.abspath(features_store))
            data.labels.to_csv(labels_store)
            self.training_data = data

    def get_data(self):
        return self.training_data


class EllipticDataProcessor:
    training_data: TrainingData
    features_file: str
    labels_file: str

    def __init__(self):
        working_dir = os.getcwd()
        self.features_file = os.path.join(working_dir, _fFEATURES_LAKE)
        self.labels_file = os.path.join(working_dir, _fLABELS_LAKE)
        self.training_data = None

    def _retrieve_data(self, features_file: str, labels_file: str) -> DataLake:
        logging.info("Reading features file")
        features_file = pd.read_csv(features_file)
        logging.info("Reading labels file")
        labels_file = pd.read_csv(labels_file)
        logging.info("Done reading files")
        return DataLake(features_file, labels_file)

    def preprocess_training_data(self) -> TrainingData:
        data_lake: DataLake
        logging.debug("Start preprocessing elliptic data")
        # Read raw data files
        data_lake = self._retrieve_data(self.features_file, self.labels_file)
        features_df = data_lake.features
        labels_df = data_lake.labels

        # Construct column names for features data
        logging.debug("Construct column headers")
        columns = [_cID, _cTIME]
        for i in range(1, len(features_df.columns) - 1):
            columns.append(f"feature_{i}")
        features_df.columns = columns

        # Rename labels ["licit": 2, "illicit": 1, "unlabeled": "unknown"]
        # to            ["licit": 0, "illicit": 1, "unlabeled": -1 ]
        logging.debug("Change label names")
        labels_df = labels_df.replace(
            {_cRESULT: {"unknown": -1, "1": 1, "2": 0}}
        ).astype({_cRESULT: int}, errors="raise")
        labels_df.loc[labels_df[_cRESULT] == 2, _cRESULT] = 0

        # Align features and labels, drop unlabeled
        logging.debug("Align features and labels")
        combined_df = pd.merge(features_df, labels_df, on=_cID, how="left")
        combined_df = combined_df[combined_df[_cRESULT] != -1]

        features_df = combined_df.drop(columns=[_cRESULT])
        features_df = combined_df.drop(columns=[_cID])
        labels_df = combined_df[_cRESULT]

        # Basic data engineering: Normalise feature data
        logging.debug("Normalise feature data")
        coltime = features_df[_cTIME]
        std_scaler = StandardScaler()
        features_df = pd.DataFrame(std_scaler.fit_transform(features_df))
        features_df[_cTIME] = coltime

        logging.debug("Done preprocessing data")
        return TrainingData(features_df, labels_df)

    def get_full_dataset(self, force_rebuild=False) -> TrainingData:
        if not self.training_data or force_rebuild:
            logging.info("Building training data")
            return self.preprocess_training_data()
        else:
            return self.training_data

