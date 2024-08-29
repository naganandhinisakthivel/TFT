import torch.nn as nn
from pytorch_forecasting import TemporalFusionTransformer

class TFTModel(nn.Module):
    def __init__(self, dataset, hidden_size=160, attention_heads=4, dropout=0.1, num_hidden_layers=8):
        super(TFTModel, self).__init__()
        self.model = TemporalFusionTransformer.from_dataset(
            dataset,
            learning_rate=[0.001, 0.0001, 0.01],
            hidden_size=hidden_size,
            attention_head_size=attention_heads,
            dropout=dropout,
            hidden_continuous_size=8,
            output_size=7,
            loss=nn.MSELoss(),
            log_interval=10,
            reduce_on_plateau_patience=4,
            optimizer="adam",
            num_hidden_layers=num_hidden_layers,
        )

    def forward(self, x):
        return self.model(x)

    def fit(self, train_dataloader, epochs=100):
        optimizer = torch.optim.Adam(self.parameters())
        for epoch in range(epochs):
            self.train()
            for batch in train_dataloader:
                optimizer.zero_grad()
                predictions = self(batch)
                loss = self.model.calculate_loss(batch, predictions)
                loss.backward()
                optimizer.step()
            print(f"Epoch {epoch + 1}/{epochs}, Loss: {loss.item()}")

    def evaluate(self, val_dataloader):
        self.eval()
        with torch.no_grad():
            for batch in val_dataloader:
                predictions = self(batch)
                loss = self.model.calculate_loss(batch, predictions)
                print(f"Evaluation Loss: {loss.item()}")
