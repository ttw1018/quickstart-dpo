import torch
from transformers import LlamaModel, LlamaPreTrainedModel, LlamaConfig
from torch import nn


class MiniLlama(nn.Module):
  def __init__(self, config: LlamaConfig):
    super().__init__()
    self.config = config
    self.model = LlamaModel(config)
    self.lm_head = nn.Linear(config.hidden_size, config.vocab_size, bias=False)

  def forward(self, input_ids, attention_mask=None, **kwargs):
    output = self.model(input_ids, attention_mask, **kwargs)[0]
    return self.lm_head(output)

  def generate(self, input_ids, max_new_tokens, eos_id):
    result = []
    for _ in range(input_ids.shape[0]):
      ids = input_ids[_].unsqueeze(0)
      for i in range(max_new_tokens):
        output = self(ids)
        new_id = output[:, -1].argmax(-1).unsqueeze(1)
        ids = torch.concat([ids, new_id], dim=-1)
        if new_id[0][0] == eos_id:
          break
      ids = ids.squeeze()
      result.append(ids.cpu().tolist())
    return result
