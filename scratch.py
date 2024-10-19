import torch
import torch.nn.functional as F
import math
from torch.optim import AdamW
from torch.utils.data import DataLoader, TensorDataset
from fancy_einsum import einsum

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.colors import LinearSegmentedColormap

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

        self.EQKE = None
        self.EQKP = None
        self.EVOU = None
        self.PVOU = None
        self.direct_path = None


    def forward(self, input_ids):
        # Note: dropping dim=0 ie batch dim
        # breakpoint()
        x_query = F.one_hot(input_ids, num_classes=d_vocab).squeeze(0).float()
        x = F.one_hot(input_ids, num_classes=d_vocab).squeeze(0).float()

        assert x_query.shape == x.shape
        # assert x_query.shape == (n_ctx, d_vocab)

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

        self.EQKE = E_q @ self.Q @ self.K.T @ E_bar.T

        assert self.EQKE.shape == (d_vocab, d_vocab)

        # P_hat = P - P_bar
        P_hat = self.P - self.P_query

        # Position dependent
        self.EQKP = E_q @ self.Q @ self.K.T @ P_hat.T
        assert self.EQKP.shape == (d_vocab, n_ctx)   

        QK = x_query @ (self.EQKE@x.T + self.EQKP)  
        assert QK.shape == (n_ctx, n_ctx)

        self.EVOU = E_bar @ self.V @ self.O @ self.U
        self.PVOU = P_hat @ self.V @ self.O @ self.U

        OV = x @ self.EVOU + self.PVOU
        assert OV.shape == (n_ctx, d_vocab)

        self.direct_path = (E_q @ self.U)
        #TODO;l assert direct_path
        # assert self.direct_path.shape == (n_ctx, d_vocab)

        # Apply causal mask
        causal_mask = torch.tril(torch.ones(n_ctx, n_ctx))
        QK = QK.masked_fill(causal_mask == 0, float('-inf'))

        M = torch.softmax(QK/math.sqrt(d_model), dim=-1) @ OV + (x_query @  self.direct_path)
        assert M.shape == (n_ctx, d_vocab)

        final_logits = M[-1, :]
        l_max = torch.argmax(final_logits)
        return M

# Generate random sequences
#num_sequences = 384000  #TODO: use full seequences
num_sequences = 3840
batch_size = 1 #TODO: use batch size 128


sequences = torch.randint(0, d_vocab, (num_sequences, n_ctx))

dataset = TensorDataset(sequences)
dataloader = DataLoader(dataset, batch_size=1, shuffle=True)

model = Transformer()
optimizer = AdamW(model.parameters(), lr=0.001, betas=(0.9, 0.999), weight_decay=0.01)

num_epochs = 1
num_steps = num_sequences // batch_size

for epoch in range(num_epochs):
    for step, (batch,) in enumerate(dataloader):
        if step >= num_steps:
            break

        optimizer.zero_grad()

        # Forward pass
        input_ids = batch
        logits = model(input_ids)

        # Compute loss
        targets = input_ids[:, -1]  # Last token in each sequence
        loss = F.cross_entropy(logits, input_ids.squeeze())

        # Backward pass and optimization
        loss.backward()
        optimizer.step()

        if step % 100 == 0:
            print(f"Epoch: {epoch+1}, Step: {step}, Loss: {loss.item():.4f}")

print("Training completed.")

# Save the trained model
torch.save(model.state_dict(), "trained_transformer.pth")

def plot_heatmap(matrix, title, xlabel, ylabel):
    # Convert tensor to numpy array if it's not already
    if hasattr(matrix, 'detach'):
        matrix = matrix.detach().numpy()
        
    # Create custom colormap
    colors = ['darkred', 'red', 'orange', 'yellow', 'white', 'lightblue', 'blue', 'darkblue']
    n_bins = 256  # Number of color gradations
    cmap = LinearSegmentedColormap.from_list("custom", colors, N=n_bins)

    # Create the plot
    plt.figure(figsize=(10, 8))
    im = plt.imshow(matrix, cmap=cmap, aspect='auto', 
                    vmin=np.percentile(matrix, 1), vmax=np.percentile(matrix, 99))
    
    # # Add colorbar
    cbar = plt.colorbar(im)
    # cbar.set_label('Attention Score')

    # Set title and labels
    plt.title(title, fontsize=16)
    plt.xlabel(xlabel, fontsize=12)
    plt.ylabel(ylabel, fontsize=12)

    # Add grid lines
    plt.grid(which='major', color='w', linestyle='-', linewidth=0.5)
    
    # Show the plot
    plt.tight_layout()
    plt.show()

plot_heatmap(model.EQKE, "EQKE (Position-Independent Attention)", "key token", "query token")
plot_heatmap(model.EQKP, "EQKP (Position-Dependent Attention)", "key position", "query token")
plot_heatmap(model.EVOU, "EVOU (Value Output)", "output logit token", "input token")
plot_heatmap(model.PVOU, "PVOU (Position-Dependent Output)", "output logit token", "input position")
plot_heatmap(model.direct_path, "Direct Path", "output logit token", "input token")