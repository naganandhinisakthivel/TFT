from data_loader import DataLoader
from model import TFTModel

def main():
    data_loader = DataLoader(file_path="path/to/your/data.csv", max_encoder_length=48, max_prediction_length=24)
    
    # Load and preprocess data
    data = data_loader.load_data()
    data = data_loader.preprocess_data(data)
    
    # Create dataset
    dataset = data_loader.create_dataset(data)
    
    # Initialize model
    model = TFTModel(dataset=dataset)
    
    # Train the model
    train_dataloader = data_loader.get_dataloader(train=True, batch_size=64)
    model.fit(train_dataloader=train_dataloader, epochs=100)
    
    # Evaluate the model
    val_dataloader = data_loader.get_dataloader(train=False, batch_size=64)
    model.evaluate(val_dataloader)

if __name__ == "__main__":
    main()
