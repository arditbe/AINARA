import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import numpy as np
import pickle


class Attention(nn.Module):
    def __init__(self, hidden_dim):
        super(Attention, self).__init__()
        self.attn = nn.Linear(hidden_dim * 2, hidden_dim)
        self.v = nn.Parameter(torch.rand(hidden_dim))

    def forward(self, hidden, encoder_outputs):
        batch_size = encoder_outputs.shape[0]
        src_len = encoder_outputs.shape[1]

        hidden = hidden[-1].unsqueeze(1).repeat(1, src_len, 1)
        energy = torch.tanh(self.attn(torch.cat((hidden, encoder_outputs), dim=2)))

        attention = torch.sum(self.v * energy, dim=2)
        return torch.softmax(attention, dim=1)


class Encoder(nn.Module):
    def __init__(self, vocab_size, embedding_dim, hidden_dim):
        super(Encoder, self).__init__()
        self.embedding = nn.Embedding(vocab_size, embedding_dim)
        self.lstm = nn.LSTM(embedding_dim, hidden_dim, batch_first=True)

    def forward(self, src):
        embedded = self.embedding(src)
        outputs, (hidden, cell) = self.lstm(embedded)
        return outputs, hidden, cell


class Decoder(nn.Module):
    def __init__(self, vocab_size, embedding_dim, hidden_dim):
        super(Decoder, self).__init__()
        self.embedding = nn.Embedding(vocab_size, embedding_dim)
        self.attention = Attention(hidden_dim)
        self.lstm = nn.LSTM(embedding_dim + hidden_dim, hidden_dim, batch_first=True)
        self.fc_out = nn.Linear(hidden_dim * 2, vocab_size)

    def forward(self, input, hidden, cell, encoder_outputs):
        input = input.unsqueeze(1)
        embedded = self.embedding(input)

        attn_weights = self.attention(hidden, encoder_outputs)
        attn_weights = attn_weights.unsqueeze(1)

        context = torch.bmm(attn_weights, encoder_outputs)
        rnn_input = torch.cat((embedded, context), dim=2)

        output, (hidden, cell) = self.lstm(rnn_input, (hidden, cell))
        prediction = self.fc_out(torch.cat((output.squeeze(1), context.squeeze(1)), dim=1))

        return prediction, hidden, cell


class Seq2Seq(nn.Module):
    def __init__(self, encoder, decoder):
        super(Seq2Seq, self).__init__()
        self.encoder = encoder
        self.decoder = decoder

    def forward(self, src, trg, teacher_forcing_ratio=0.5):
        batch_size = trg.shape[0]
        trg_len = trg.shape[1]
        vocab_size = self.decoder.fc_out.out_features
        outputs = torch.zeros(batch_size, trg_len, vocab_size).to(src.device)

        encoder_outputs, hidden, cell = self.encoder(src)
        input = trg[:, 0]

        for t in range(1, trg_len):
            output, hidden, cell = self.decoder(input, hidden, cell, encoder_outputs)
            outputs[:, t, :] = output
            top1 = output.argmax(1)
            input = trg[:, t] if torch.rand(1).item() < teacher_forcing_ratio else top1

        return outputs


def load_ai_model():
    with open("tokenizer.pickle", "rb") as f:
        vocab = pickle.load(f)

    vocab_size = len(vocab)
    embedding_dim = 256
    hidden_dim = 512

    encoder = Encoder(vocab_size, embedding_dim, hidden_dim)
    decoder = Decoder(vocab_size, embedding_dim, hidden_dim)
    seq2seq_model = Seq2Seq(encoder, decoder)

    state_dict = torch.load("friday_model.pt", map_location=torch.device("cpu"))
    seq2seq_model.load_state_dict(state_dict)

    seq2seq_model.eval()
    return seq2seq_model, vocab


def predict(model, vocab, input_text):
    input_tokens = [vocab.get(word, vocab["<UNK>"]) for word in input_text.split()]
    input_tensor = torch.tensor(input_tokens).unsqueeze(0)

    encoder_outputs, hidden, cell = model.encoder(input_tensor)

    input_token = torch.tensor([vocab["<SOS>"]])
    predicted_words = []

    for _ in range(20):
        output, hidden, cell = model.decoder(input_token, hidden, cell, encoder_outputs)
        predicted_token = output.argmax(1).item()

        if predicted_token == vocab["<EOS>"]:
            break

        predicted_words.append(predicted_token)
        input_token = torch.tensor([predicted_token])

    reversed_vocab = {v: k for k, v in vocab.items()}
    return " ".join(reversed_vocab.get(token, "<UNK>") for token in predicted_words)


if __name__ == "__main__":
    import sys
    if len(sys.argv) < 2:
        print("Usage: python fetch_ai.py <input_text>")
        sys.exit(1)

    input_text = sys.argv[1]
    model, vocab = load_ai_model()
    response = predict(model, vocab, input_text)
    print(response)
