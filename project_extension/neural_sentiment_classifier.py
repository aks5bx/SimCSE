import torch
import torch.nn as nn


class NeuralSentimentClassifier(nn.Module):

    def __init__(self, input_dim: int, hidden_dim: int, n_layers: int, n_classes=3):
        super(NeuralSentimentClassifier, self).__init__()

        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        
        self.layers = [nn.Linear(self.input_dim, self.hidden_dim), nn.ReLU()]
        for i in range(n_layers):
            self.layers += [nn.Linear(self.hidden_dim, self.hidden_dim), nn.ReLU()]
        self.layers += [nn.Linear(self.hidden_dim, n_classes), nn.Softmax(dim=1)]

        self.model = nn.Sequential(*self.layers)
    
    def forward(self, sent_embeddings) -> torch.Tensor:
        """Takes a 2D tensor of sentence embeddings and returns a tensor of class probabilities."""

        output = self.model(sent_embeddings)
        return output

