import torch
class Model(torch.nn.Module):
    def __init__(self, vocab_size, ndim=384):
        super().__init__()
        self.center_embed = torch.nn.Embedding(num_embeddings=vocab_size, embedding_dim=ndim)
        self.context_embed = torch.nn.Embedding(num_embeddings=vocab_size, embedding_dim=ndim)

        torch.nn.init.xavier_uniform_(self.center_embed.weight)
        torch.nn.init.xavier_uniform_(self.context_embed.weight)

    def forward(self, batch):
        # batch should be a tuple (centers, contexts_and_negatives), each should be a torch.tensor

        centers = batch[0]
        contexts_and_negatives = batch[1]

        v = self.center_embed(centers)
        u = self.context_embed(contexts_and_negatives)

        batch_size = len(centers)
        v = v.reshape((batch_size, 1, -1))

        return torch.bmm(v, u.permute(0, 2, 1))