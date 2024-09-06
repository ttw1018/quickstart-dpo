import torch
from tokenizer import Tokenizer
from utils import load_model, load_batch, load_test
from tqdm import tqdm
from matplotlib import pyplot as plt


@torch.no_grad()
def test(model, tokenizer):
  split = {}
  error_list = []
  for i in range(100):
    data = load_test(1)
    input_ids, attention_mask, _ = tokenizer.encode(data)
    output = model.generate(
      input_ids,
      max_new_tokens=32,
      eos_id=tokenizer.vocab2id["E"],
    )
    predict = tokenizer.decode(output)[0]

    try:
      assert predict[-1] == "E" and predict.count("=") == 1
      predict = predict[:-1]
      p1, p2 = predict.split("=")
      v = eval(p2)
      error_list.append(abs(int(p1) - v))
      if (p2.count("+") + 1) not in split:
        split[p2.count("+") + 1] = 0
      split[p2.count("+") + 1] = split[p2.count("+") + 1] + 1
    except Exception as _:
      pass
  print(split)
  print(sum(error_list) / len(error_list))
  print(f"acc: {len(error_list)}")

  plt.figure()
  plt.bar(list(split.keys()), list(split.values()))
  plt.title(
    f"average error: {sum(error_list) / len(error_list)}\n following instruct: {len(error_list) / 100:.2f}"
  )
  plt.savefig("sft-dist.jpg")


def train():
  steps = 5000
  model = load_model()
  optimizer = torch.optim.Adam(model.parameters(), 1e-4)
  scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, steps)
  loss_fn = torch.nn.CrossEntropyLoss()
  tokenizer = Tokenizer()

  loss_list = []

  pbar = tqdm(range(steps))

  for i in pbar:
    scheduler.step()
    model.train()
    data = load_batch(256)
    input_ids, attention_mask, labels = tokenizer.encode(data, True)
    output = model(input_ids, attention_mask)
    loss = loss_fn(output[:, :-1].flatten(0, 1), labels[:, 1:].flatten())
    loss_list.append(loss.item())
    loss.backward()
    optimizer.step()
    optimizer.zero_grad()
    pbar.set_description(f"loss:{loss.item():.4f}")

  torch.save(model.state_dict(), "sft.bin")
  plt.figure()
  plt.plot(range(len(loss_list)), loss_list)
  plt.savefig("sft-loss.jpg")

  test(model, tokenizer)


if __name__ == "__main__":
  train()
