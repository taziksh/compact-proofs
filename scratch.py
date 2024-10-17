import torch
import torch.nn.functional as F
import math

d_vocab = 64
d = d_head = d_model = 32
n_ctx = k = 4

t = torch.randint(low=0, high=d_vocab, size=(n_ctx,))

# why is (n_ctx,) != (n_ctx)
assert t.shape == (n_ctx,)

class Transformer(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.E = torch.nn.Parameter(torch.nn.init.xavier_uniform_(torch.empty(d_vocab, d_model)))
        self.P = torch.nn.Parameter(torch.nn.init.xavier_uniform_(torch.empty(n_ctx, d_model)))
        self.P_query = torch.nn.Parameter(torch.nn.init.xavier_uniform_(torch.empty(n_ctx, d_model)))


        self.Q = torch.nn.Parameter(torch.nn.init.xavier_uniform_(torch.empty(d_model, d_model)))
        self.K = torch.nn.Parameter(torch.nn.init.xavier_uniform_(torch.empty(d_model, d_model)))
        self.V = torch.nn.Parameter(torch.nn.init.xavier_uniform_(torch.empty(d_model, d_model)))
        self.O = torch.nn.Parameter(torch.nn.init.xavier_uniform_(torch.empty(d_model, d_model)))

        self.U = torch.nn.Parameter(torch.nn.init.xavier_uniform_(torch.empty(d_model, d_vocab)))


    def forward(self, input_ids, attention_mask=None):
        # Note: dropping dim=0 ie batch dim
        x_query = F.one_hot(input_ids, num_classes=d_vocab).squeeze(0).float()
        x = F.one_hot(input_ids, num_classes=d_vocab).squeeze(0).float()

        assert x_query.shape == x.shape
        assert x_query.shape == (n_ctx, d_vocab)

        # dim=-1 => summing across d_model dimension
        P_avg = torch.sum(self.P) / n_ctx
        
        # assert P_avg.shape == (1,)

        # E_bar = E + P_bar
        E_bar = self.E + P_avg.unsqueeze(0).expand(d_vocab, d_model)
        assert E_bar.shape == (d_vocab, d_model)

        # E_q = E + P_q
        # Mean since we want it to be position independent
        E_q = self.E + self.P_query.mean(dim=0, keepdim=True).expand(d_vocab, d_model)
        assert E_q.shape == (d_vocab, d_model)

        # Position independent
        EQKE = E_q @ self.Q @ self.K.T @ E_bar.T

        assert EQKE.shape == (d_vocab, d_vocab)

        # P_hat = P - P_bar
        P_hat = self.P - self.P_query

        # Position dependent
        EQKP = E_q @ self.Q @ self.K.T @ P_hat.T
        assert EQKP.shape == (d_vocab, n_ctx)   

        QK = x_query @ (EQKE@x.T + EQKP)  
        print(QK.shape)
        assert QK.shape == (n_ctx, n_ctx)

        EVOU = E_bar @ self.V @ self.O @ self.U
        PVOU = P_hat @ self.V @ self.O @ self.U

        OV = x @ EVOU + PVOU
        print(OV.shape)
        assert OV.shape == (n_ctx, d_vocab)

        direct_path = x_query @ (E_q @ self.U)
        assert direct_path.shape == (n_ctx, d_vocab)

        # TODO: is this dot product??
        # TODO: (seq_len, seq_len) but which do I softmax over?
        M = torch.softmax(QK/math.sqrt(d_model), dim=-1) @ OV + direct_path
        assert M.shape == (n_ctx, d_vocab)

        # TODO: i think
        final_logits = M[-1, :]
        l_max = torch.argmax(final_logits)
        print(l_max)

        



    
model = Transformer()
# Create dummy inputs
dummy_input_ids = torch.randint(0, d_vocab, (1, n_ctx))  # Batch size of 1
dummy_attention_mask = torch.ones(1, n_ctx)  # Full attention for all tokens

# Invoke the model
with torch.no_grad():
    output = model(dummy_input_ids, dummy_attention_mask)

