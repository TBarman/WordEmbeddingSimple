import torch
from torch.utils.data import DataLoader, TensorDataset
from torch.nn.functional import binary_cross_entropy_with_logits
import time
class Trainer:
    def __init__(self, model, data, batch_size=32, lr=0.01):
        self.batch_size = batch_size
        self.lr = lr
        self.data = data

        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        print(f"Using device: {self.device}")
        self.model = model.to(self.device)

        tensor_dataset = TensorDataset(*data)
        self.dataloader = DataLoader(tensor_dataset, self.batch_size, shuffle=True, num_workers=2)

        self.optimizer = torch.optim.Adam(model.parameters(), lr=lr)

        neg_cols = data[1].shape[1]
        self.label_row = torch.tensor([1.0] + [0.0] * (neg_cols - 1)).to(self.device)

    def fit(self, epochs=10):
        print("Training Started...")
        for i in range(epochs):
            start = time.time()
            avg_loss = self.fit_epoch()
            print(f"Epoch {i+1}: Time = {time.time() - start:.2f}s, Loss = {avg_loss:.4f}")

    def fit_epoch(self):
        total_loss = 0.0
        batches = 0
        for batch in self.dataloader:
            batch = [t.to(self.device) for t in batch]
            current_batch_size = batch[0].shape[0]
            self.optimizer.zero_grad()
            logits = self.model(batch).squeeze(1)
            labels = self.label_row.unsqueeze(0).expand(current_batch_size, -1)
            loss = binary_cross_entropy_with_logits(logits, labels)
            loss.backward()
            self.optimizer.step()

            total_loss += loss.item()
            batches += 1

        return total_loss / batches


