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

  print(sum(error_list) / len(error_list))
  print(f"acc: {len(error_list)}")

  plt.figure()
  plt.bar(list(split.keys()), list(split.values()))
  plt.title(
    f"average error: {sum(error_list) / len(error_list)}\n following instruct: {len(error_list) / 100:.2f}"
  )
  plt.savefig("dpo-dist.jpg")


def pad(x, pad_len, dim, token_id):
  return torch.concat(
    (torch.full((x.shape[0], pad_len - x.shape[1]), token_id).to("cuda"), x), dim=dim
  )


def cal_prob(model, select, reject, pad_token_id, eos_token_id):
  b = select[0].shape[0]
  max_len = max(select[0].shape[1], reject[0].shape[1])

  input_ids = torch.concat(
    [
      pad(select[0], max_len, 1, pad_token_id),
      pad(reject[0], max_len, 1, pad_token_id),
    ],
    dim=0,
  )

  attention_mask = torch.concat(
    [
      pad(select[1], max_len, 1, pad_token_id),
      pad(reject[1], max_len, 1, pad_token_id),
    ],
    dim=0,
  )

  labels = torch.concat(
    [
      pad(select[2], max_len, 1, pad_token_id),
      pad(reject[2], max_len, 1, pad_token_id),
    ],
    dim=0,
  )

  logits = model(input_ids, attention_mask)

  logits = logits[:, :-1].softmax(-1).log()

  labels = labels[:, 1:]

  labels[labels == -100] = 0
  labels[labels == pad_token_id] = 0
  # labels[labels == eos_token_id] = 0
  mask = labels.clone()
  mask = mask != 0
  labels = labels.unsqueeze(2)

  prob = (logits.gather(2, labels).squeeze(2) * mask).sum(-1)

  return prob[:b], prob[b:]


def run():
  model = load_model("sft.bin")
  ref = load_model("sft.bin")

  optimizer = torch.optim.Adam(model.parameters(), lr=1e-5)

  tokenizer = Tokenizer()

  pbar = tqdm(range(200))

  loss_list = []

  for i in pbar:
    select, reject = load_batch(256, 3)
    select = tokenizer.encode(select, True)
    reject = tokenizer.encode(reject, True)

    prob_select, prob_reject = cal_prob(
      model, select, reject, tokenizer.vocab2id["P"], tokenizer.vocab2id["E"]
    )

    with torch.no_grad():
      prob_ref_select, prob_ref_reject = cal_prob(
        ref, select, reject, tokenizer.vocab2id["P"], tokenizer.vocab2id["E"]
      )

    prob1 = prob_select - prob_ref_select
    prob2 = prob_reject - prob_ref_reject

    prob = (prob1 - prob2) * 0.8

    loss = -prob.sigmoid().log().mean()

    loss_list.append(loss.item())

    pbar.set_description(f"loss: {loss.item()}")

    loss.backward()
    optimizer.step()
    optimizer.zero_grad()

  plt.figure()
  plt.plot(range(len(loss_list)), loss_list)
  plt.savefig("dop-loss.jpg")

  test(model, tokenizer)


if __name__ == "__main__":
  run()
