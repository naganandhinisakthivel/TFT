# evaluate.py

from data_loader import DataLoader
from model import TFTModel
from config import Config
from utils import load_model

def evaluate_model():
    # Load and preprocess the data
    data_loader = DataLoader(file_path=Config.file_path, max_encoder_length=Config.max_encoder_length, max_prediction_length=Config.max_prediction_length)
    data = data_loader.load_data()
    data = data_loader.preprocess_data(data)
    
    # Create dataset
    dataset = data_loader.create_dataset(data)
    val_dataloader = data_loader.get_dataloader(train=False, batch_size=Config.batch_size)
    
    # Initialize and load the model
    model = TFTModel(dataset=dataset)
    model = load_model(model, "path/to/saved_model.pth")
    
    # Evaluate the model
    model.evaluate(val_dataloader)

if __name__ == "__main__":
    evaluate_model()
