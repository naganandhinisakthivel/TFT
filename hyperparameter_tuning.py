# hyperparameter_tuning.py

from data_loader import DataLoader
from model import TFTModel
from config import Config
from sklearn.model_selection import ParameterGrid

def tune_hyperparameters():
    # Define the grid of hyperparameters to search
    param_grid = {
        'hidden_size': [128, 160, 200],
        'attention_heads': [2, 4, 8],
        'dropout': [0.1, 0.2, 0.3],
        'num_hidden_layers': [4, 8, 12]
    }
    
    # Load and preprocess data
    data_loader = DataLoader(file_path=Config.file_path, max_encoder_length=Config.max_encoder_length, max_prediction_length=Config.max_prediction_length)
    data = data_loader.load_data()
    data = data_loader.preprocess_data(data)
    dataset = data_loader.create_dataset(data)
    
    # Tune model
    best_loss = float('inf')
    best_params = {}
    for params in ParameterGrid(param_grid):
        print(f"Testing parameters: {params}")
        model = TFTModel(dataset=dataset, **params)
        train_dataloader = data_loader.get_dataloader(train=True, batch_size=Config.batch_size)
        model.fit(train_dataloader, epochs=Config.epochs)
        
        # Evaluate model performance
        val_dataloader = data_loader.get_dataloader(train=False, batch_size=Config.batch_size)
        loss = model.evaluate(val_dataloader)
        
        if loss < best_loss:
            best_loss = loss
            best_params = params
    
    print(f"Best parameters: {best_params}")
    print(f"Best loss: {best_loss}")

if __name__ == "__main__":
    tune_hyperparameters()
