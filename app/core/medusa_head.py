import torch


class MedusaHead(torch.nn.Module):
    def __init__(self, base_model, num_heads=3, device=None):
        super().__init__()
        self.base_model = base_model
        self.num_heads = num_heads
        self.device = device or torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.heads = torch.nn.ModuleList([
            torch.nn.Linear(base_model.n_embd, base_model.n_vocab).to(self.device)
            for _ in range(num_heads)
        ])

    def forward(self, hidden_states):
        base_logits = torch.tensor(self.base_model.eval(hidden_states.cpu().tolist(), echo=False)['logits']).to(self.device)
        medusa_logits = [head(hidden_states) for head in self.heads]
        return base_logits, medusa_logits

    def generate(self, prompt, max_tokens=100):
        input_ids = torch.tensor(self.base_model.tokenize(prompt)).to(self.device)
        for _ in range(max_tokens):
            hidden_states = torch.tensor(self.base_model.eval(input_ids.cpu().tolist(), echo=True)['hidden_states'][-1]).to(self.device)
            base_logits, medusa_logits = self.forward(hidden_states.unsqueeze(0))
            
            combined_logits = base_logits + sum(medusa_logits)
            next_token = torch.argmax(combined_logits, dim=-1)
            
            input_ids = torch.cat([input_ids, next_token])
            
            if next_token.item() == self.base_model.token_eos():
                break
        
        return self.base_model.detokenize(input_ids.cpu().detach().tolist())