from random import randint
import torch
from transformers import LlamaConfig
from model import MiniLlama


def load_batch(batch_size, num=None):
  select = []
  reject = []
  for i in range(batch_size):
    nums = []
    cnt = randint(2, 5)
    while cnt == num:
      cnt = randint(2, 5)
    for _ in range(cnt):
      nums.append(randint(-50, 50))
    reject.append(f"{sum(nums)}={'+'.join(map(str, nums))}E")

    if num:
      nums1 = []
      for _ in range(num - 1):
        nums1.append(randint(-50, 50))
      nums1.append(sum(nums) - sum(nums1))
      select.append(f"{sum(nums1)}={'+'.join(map(str, nums1))}E")

  if num:
    return select, reject
  else:
    return reject


def load_test(batch_size):
  data = []
  for i in range(batch_size):
    data.append(f"{randint(-200, 200)}=")
  return data


def load_model(checkpoint=None):
  config = LlamaConfig(
    hidden_size=64,
    intermediate_size=128,
    max_position_embeddings=64,
    num_attention_heads=8,
    num_hidden_layers=4,
    num_key_value_heads=4,
    vocab_size=15,
  )
  model = MiniLlama(config)
  if checkpoint:
    model.load_state_dict(torch.load(checkpoint))
  device = "cuda" if torch.cuda.is_available() else "cpu"
  model.to(device)
  return model


if __name__ == "__main__":
  print(load_batch(10, 1))
  print(load_model())
