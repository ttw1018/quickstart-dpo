import torch


class Tokenizer:
  def __init__(self):
    self.vocab = list("0123456789+-=EP")
    self.vocab2id = {}
    self.id2vocab = {}
    for k, v in enumerate(self.vocab):
      self.vocab2id[v] = k
      self.id2vocab[k] = v
    self.device = "cuda" if torch.cuda.is_available() else "cpu"

  def encode(self, input, need_label=False):
    input_ids = [[self.vocab2id[v] for v in i] for i in input]

    max_len = max(len(i) for i in input_ids)

    input_ids = [[self.vocab2id["P"]] * (max_len - len(i)) + i for i in input_ids]

    if need_label:
      label_index = [i.index(self.vocab2id["="]) for i in input_ids]

    input_ids = torch.tensor(input_ids).to(self.device)
    mask = (input_ids != self.vocab2id["P"]).long().to(self.device)

    label = None

    if need_label:
      label = input_ids.clone()
      for i in range(len(label_index)):
        label[i, : label_index[i] + 1] = -100
      label.to(self.device)
    return input_ids, mask, label

  def decode(self, output):
    tokens = ["".join([self.id2vocab[j] for j in i]) for i in output]
    return tokens


if __name__ == "__main__":
  tokenizer = Tokenizer()
  print(tokenizer.encode(["1+8=2", "3+5+6+1345234"]))

  print(tokenizer.decode(tokenizer.encode(["1+8=2", "3+5+6+1345234"])))
