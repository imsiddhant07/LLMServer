import torch
import torch.nn as nn

class MedusaHead(nn.Module):
    def __init__(self, base_model, num_heads=3):
        super().__init__()
        self.base_model = base_model
        self.num_heads = num_heads
        self.heads = nn.ModuleList([nn.Linear(base_model.config.hidden_size, base_model.config.vocab_size) for _ in range(num_heads)])

    def forward(self, hidden_states):
        base_logits = self.base_model.lm_head(hidden_states)
        medusa_logits = [head(hidden_states) for head in self.heads]
        return base_logits, medusa_logits

class MedusaModel:
    def __init__(self, base_model):
        self.medusa = MedusaHead(base_model)

    def generate(self, text, max_tokens=100):
        input_ids = self.base_model.tokenizer.encode(text, return_tensors="pt")
        output_ids = input_ids.clone()

        for _ in range(max_tokens):
            hidden_states = self.base_model.model(output_ids)['last_hidden_state'][:, -1:]
            base_logits, medusa_logits = self.medusa(hidden_states)

            # Implement speculative decoding logic here
            # This is a simplified version and needs to be expanded
            next_token_id = torch.argmax(base_logits, dim=-1)
            output_ids = torch.cat([output_ids, next_token_id], dim=-1)

        return self.base_model.tokenizer.decode(output_ids[0])
