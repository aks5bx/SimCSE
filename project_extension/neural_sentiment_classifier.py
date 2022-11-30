import torch
import torch.nn as nn
import numpy as np


class NeuralNet(nn.Module):

    def __init__(self, sent_embeddings, hidden_dim: int):
        super(NeuralNet, self).__init__()

        self.sent_embeddings = sent_embeddings # batch_size x emb_dim
        self.input_dim = sent_embeddings.size(-1)
        self.hidden_dim = hidden_dim
        
        self.model = nn.Sequential(
            nn.Linear(self.input_dim, self.hidden_dim),
            nn.ReLU(),
            nn.Linear(self.hidden_dim, 1),
            nn.Softmax()
        )
    
    def forward(self, sent_embeddings) -> torch.Tensor:
        """Takes a 2D tensor of sentence embeddings and returns a tensor of class probabilities."""

        output = self.model(sent_embeddings)
        return output
        

class NeuralSentimentClassifier(object):
    """Runs NeuralNet to generate sentiment probability scores and predictions for sentence embeddings"""
    def __init__(self, sent_embeddings, hidden_dim: int):
        self.sent_embeddings = sent_embeddings
        self.hidden_dim = hidden_dim
        self.neural_net = NeuralNet(self.sent_embeddings, self.hidden_dim)

    def predict(self) -> int:
        probs = self.neural_net.forward(self.sent_embeddings)
        return probs, torch.argmax(probs)