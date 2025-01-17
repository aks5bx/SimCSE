import torch
from torch.utils.data import Dataset

class TweetDataset(Dataset):
    """
    Class that represents a train/validation/test dataset that's readable for PyTorch
    Note that this class inherits torch.utils.data.Dataset
    """

    def __init__(self, data):
        self.sentences = data['sentences']
        self.sent_labels = data['sent_labels']
        
    def __len__(self):
        return len(self.sent_labels)

    def __getitem__(self, idx):
        """
        Triggered when you call dataset[i]
        """

        return (self.sentences[idx], self.sent_labels[idx])


def get_sent_embeddings(tokenizer, encoder, sentences, device):
    inputs = tokenizer(sentences, padding=True, truncation=True, return_tensors='pt').to(device)

    with torch.no_grad():
        embeddings = encoder(**inputs, output_hidden_states=True, return_dict=True).pooler_output
    
    return embeddings


def tweet_batch_collate(batch, tokenizer1, encoder1, tokenizer2, encoder2, device, permute1=False, permute2=False):
    sentences = [b[0] for b in batch]
    labels = torch.LongTensor([b[1] for b in batch])

    embeddings = get_sent_embeddings(tokenizer1, encoder1, sentences, device)

    if permute1:
        embeddings = embeddings[torch.randperm(embeddings.size()[0])]

    if tokenizer2 and encoder2 and not permute2:
        embeddings2 = get_sent_embeddings(tokenizer2, encoder2, sentences, device)
        embeddings = torch.cat((embeddings, embeddings2), dim=1)
    elif tokenizer2 and encoder2 and permute2:
        embeddings2 = get_sent_embeddings(tokenizer2, encoder2, sentences, device)
        embeddings2 = embeddings2[torch.randperm(embeddings2.size()[0])]
        embeddings = torch.cat((embeddings, embeddings2), dim=1)

    return (embeddings, labels, sentences)

