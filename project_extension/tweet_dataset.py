import torch
from torch.utils.data import Dataset

class TweetDataset(Dataset):
    """
    Class that represents a train/validation/test dataset that's readable for PyTorch
    Note that this class inherits torch.utils.data.Dataset
    """

    def __init__(self, data):
        self.embeddings = data['embeddings']
        self.sent_labels = data['sent_labels']
        
    def __len__(self):
        return len(self.sent_labels)

    def __getitem__(self, idx):
        """
        Triggered when you call dataset[i]
        """

        return (self.embeddings[idx], self.sent_labels[idx])


def get_sent_embeddings(tokenizer1, encoder1, tokenizer2, encoder2, device, sentences):
    inputs1 = tokenizer1(sentences, padding=True, truncation=True, return_tensors='pt').to(device)
    with torch.no_grad():
        embeddings = encoder1(**inputs1, output_hidden_states=True, return_dict=True).pooler_output
    
    if tokenizer2 and encoder2:
        inputs2 = tokenizer2(sentences, padding=True, truncation=True, return_tensors='pt').to(device)
        embeddings2 = encoder2(**inputs2, output_hidden_states=True, return_dict=True).pooler_output

        embeddings = torch.cat((embeddings, embeddings2), dim=1)
    
    return embeddings

