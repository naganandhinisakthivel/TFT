import pandas as pd
from pytorch_forecasting.data import TimeSeriesDataSet

class DataLoader:
    def __init__(self, file_path, max_encoder_length, max_prediction_length):
        self.file_path = file_path
        self.max_encoder_length = max_encoder_length
        self.max_prediction_length = max_prediction_length
        self.dataset = None

    def load_data(self):
        # Load your dataset here (example with CSV)
        data = pd.read_csv(self.file_path)
        return data

    def preprocess_data(self, data):
        # Example preprocessing
        data['time_idx'] = pd.to_datetime(data['timestamp']).astype(int) // 10**9
        return data

    def create_dataset(self, data):
        self.dataset = TimeSeriesDataSet(
            data,
            time_idx="time_idx",
            target="target_variable",
            group_ids=["group_id"],
            min_encoder_length=self.max_encoder_length // 2,
            max_encoder_length=self.max_encoder_length,
            min_prediction_length=1,
            max_prediction_length=self.max_prediction_length,
            static_categoricals=["static_feature"],
            time_varying_known_reals=["time_varying_real1", "time_varying_real2"],
            time_varying_unknown_reals=["target_variable"],
            add_relative_time_idx=True,
            add_target_scales=True,
        )
        return self.dataset

    def get_dataloader(self, train=True, batch_size=64):
        return self.dataset.to_dataloader(train=train, batch_size=batch_size)
