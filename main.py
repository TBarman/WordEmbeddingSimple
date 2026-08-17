from data_preprocessing import preprocess_data
from model import Model
from trainer import Trainer
import torch
import pickle

if __name__ == "__main__":
    print("Started...")
    vocab, data = preprocess_data()
    print("Data Processed...")
    model = Model(len(vocab))
    print("Model Initialized...")
    print("--------------------")
    trainer = Trainer(model, data, batch_size=2048, lr=0.0005)
    trainer.fit(epochs=10)

    torch.save(model.state_dict(), "word2vec_model.pt")
    with open("vocab.pkl", "wb") as f:
        pickle.dump(vocab, f)

    print("Saved model and vocab.")