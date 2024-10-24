import torch
import torch.nn.functional as F
import math
import random
from torch.optim import AdamW
from torch.utils.data import DataLoader, TensorDataset
from fancy_einsum import einsum

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.colors import LinearSegmentedColormap

d_vocab = 64
d = d_head = d_model = 32
n_ctx = k = 4
batch_size = 128

#Global seed
global_seed = 123
torch.manual_seed(global_seed)
np.random.seed(global_seed)
random.seed(global_seed)


def plot_heatmap(matrix, title, xlabel, ylabel):
    if matrix.dim() == 3:  # Assuming shape (batch_size, n_ctx, n_ctx)
        matrix = matrix.mean(dim=0)  # Average over batch size

    # Convert tensor to numpy array if it's not already
    if hasattr(matrix, 'detach'):
        matrix = matrix.detach().numpy()
        
    # Create custom colormap
    colors = ['darkred', 'red', 'orange', 'yellow', 'white', 'lightblue', 'blue', 'darkblue']
    n_bins = 256  # Number of color gradations
    cmap = LinearSegmentedColormap.from_list("custom", colors, N=n_bins)

    # Create the plot
    fig, ax = plt.subplots(figsize=(10, 8))
    im = ax.imshow(matrix, cmap=cmap, aspect='auto', 
                    vmin=np.percentile(matrix, 1), vmax=np.percentile(matrix, 99))
    
    cbar = plt.colorbar(im)

    ax.set_title(title, fontsize=16)
    ax.set_xlabel(xlabel, fontsize=12)
    ax.set_ylabel(ylabel, fontsize=12)
    ax.grid(which='major', color='w', linestyle='-', linewidth=0.5)

    return fig

class Transformer(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.E = torch.nn.Parameter(torch.nn.init.xavier_uniform_(torch.empty(batch_size, d_vocab, d_model)))
        self.P = torch.nn.Parameter(torch.nn.init.xavier_uniform_(torch.empty(n_ctx, d_model)))

        self.Q = torch.nn.Parameter(torch.nn.init.xavier_uniform_(torch.empty(d_model, d_model)))
        self.K = torch.nn.Parameter(torch.nn.init.xavier_uniform_(torch.empty(d_model, d_model)))
        self.V = torch.nn.Parameter(torch.nn.init.xavier_uniform_(torch.empty(d_model, d_model)))
        self.O = torch.nn.Parameter(torch.nn.init.xavier_uniform_(torch.empty(d_model, d_model)))

        self.U = torch.nn.Parameter(torch.nn.init.xavier_uniform_(torch.empty(d_model, d_vocab)))

        self.EQKE = None
        self.EQKP = None
        self.EVOU = None
        self.PVOU = None
        self.EU = None


    def forward(self, input_ids):
        # Note: dropping dim=0 ie batch dim
        x = F.one_hot(input_ids, num_classes=d_vocab).float()
        assert x.shape == (batch_size, n_ctx, d_vocab)

        P_query = self.P[-1, :]

        # dim=-1 => summing across d_model dimension
        P_bar = torch.mean(self.P, dim=0, keepdim=True)  

        assert P_bar.shape == (1, d_model) 

        # Note: P_bar is broadcasted P_bar

        # P_hat = P - P_bar
        P_hat = self.P - P_bar # (n_ctx, d_model)
        assert P_hat.shape == (n_ctx, d_model)

        # E_bar = E + P_bar
        E_bar = self.E + P_bar

        assert E_bar.shape == (batch_size, d_vocab, d_model)

        # E_q = E + P_q
        P_q = P_query.unsqueeze(0).expand(d_vocab, d_model)
        E_q = self.E + P_q
        assert E_q.shape == (batch_size, d_vocab, d_model)

        scaling_factor = 1 / math.sqrt(d_model)


        # Position independent
        #self.EQKE = E_q @ self.Q @ self.K.T @ E_bar.transpose(1, 2)
        self.EQKE = einsum('batch d_vocab d_model, d_model d_model_2 -> batch d_vocab d_model_2', E_q, self.Q)
        self.EQKE = einsum('batch d_vocab d_model_2, d_model d_model_2 -> batch d_vocab d_model', self.EQKE, self.K) # TODO: be careful with the dim order for transpose!
        self.EQKE = einsum('batch d_vocab d_model, batch d_vocab_2 d_model -> batch d_vocab d_vocab_2', self.EQKE, E_bar)

        assert self.EQKE.shape == (batch_size, d_vocab, d_vocab)

        # Position dependent
        # self.EQKP = scaling_factor * (E_q @ self.Q @ self.K.T @ P_hat.T)
        # self.EQKP = E_q @ self.Q @ self.K.T @ P_hat.T
        self.EQKP = einsum('batch d_vocab d_model, d_model d_model_2 -> batch d_vocab d_model_2', E_q, self.Q)
        self.EQKP = einsum('batch d_vocab d_model_2, d_model d_model_2 -> batch d_vocab d_model', self.EQKP, self.K)
        self.EQKP = einsum('batch d_vocab d_model, n_ctx d_model -> batch d_vocab n_ctx', self.EQKP, P_hat)

        assert self.EQKP.shape == (batch_size, d_vocab, n_ctx) 
  

        #QK = x_query @ (self.EQKE @ x.transpose(1, 2) + self.EQKP)
        #QK = einsum('batch d_vocab, batch d_vocab n_ctx -> batch d_vocab n_ctx', x_query, self.EQKE @ x.transpose(1, 2) + self.EQKP)
        QK = einsum('batch d_vocab d_vocab_2, batch n_ctx d_vocab_2 -> batch d_vocab n_ctx', self.EQKE, x) + self.EQKP
        QK = einsum('batch n_ctx_2 d_vocab, batch_size d_vocab n_ctx -> batch n_ctx_2 n_ctx', x, QK)
        # assert QK.shape == (n_ctx, n_ctx)
        # assert QK.shape == (batch_size, d_vocab, n_ctx)
        assert QK.shape == (batch_size, n_ctx, n_ctx)

        # Apply causal attention mask
        causal_mask = torch.tril(torch.ones(n_ctx, n_ctx))  # Creates n_ctx x n_ctx lower triangular matrix
        # causal_mask = causal_mask.  # Add batch dim: (1, n_ctx, n_ctx)
        QK = QK.masked_fill(~causal_mask.bool(), float('-inf'))

        QK = QK[:, -1, :]

        assert QK.shape == (batch_size, n_ctx)

        #TODO: do og QK, OV for comparison
        # out = einsum('batch d_vocab, batch d_vocab d_model ->')
        # og_QK = (x_query@self.E + self.P_query)@self.Q@self.K.T@(x@self.E+self.P).T

        #TODO: einsum these mfs

        # self.EVOU = E_bar @ self.V @ self.O @ self.U

        self.EVOU = einsum('batch d_vocab d_model, d_model d_model_2 -> batch d_vocab d_model_2', E_bar, self.V)
        self.EVOU = einsum('batch d_vocab d_model_2, d_model_2 d_model -> batch d_vocab d_model', self.EVOU, self.O)
        self.EVOU = einsum('batch d_vocab d_model, d_model d_vocab_2 -> batch d_vocab d_vocab_2', self.EVOU, self.U)

        assert self.EVOU.shape == (batch_size, d_vocab, d_vocab)

        self.PVOU = einsum('n_ctx d_model, d_model d_model_2 -> n_ctx d_model_2', P_hat, self.V)
        self.PVOU = einsum('n_ctx d_model_2, d_model_2 d_model -> n_ctx d_model', self.PVOU, self.O)
        self.PVOU = einsum('n_ctx d_model, d_model d_vocab -> n_ctx d_vocab', self.PVOU, self.U)

        assert self.PVOU.shape == (n_ctx, d_vocab)

        OV = einsum('batch n_ctx d_vocab, batch d_vocab d_vocab_2 -> batch n_ctx d_vocab_2', x, self.EVOU) + self.PVOU

        assert OV.shape == (batch_size, n_ctx, d_vocab)

        x_query = x[:, -1, :] # (batch_size, d_vocab)
        # assert x_query.shape == (batch_size, d_vocab)

        self.EU = einsum('batch d_vocab d_model, d_model d_vocab_2 -> batch d_vocab d_vocab_2', E_q, self.U)
        # assert self.direct_path.shape == (n_ctx, d_vocab)
        direct_path = einsum('batch d_vocab, batch d_vocab d_vocab_2 -> batch d_vocab_2', x_query, self.EU)
        assert direct_path.shape == (batch_size, d_vocab)

        M = einsum('batch n_ctx, batch n_ctx d_vocab -> batch d_vocab', torch.softmax(QK/scaling_factor, dim=-1), OV) + direct_path
        #assert M.shape == (n_ctx, d_vocab)
        assert M.shape == (batch_size, d_vocab)

        # Calculate mean values of all tensors
        tensor_means = {
            # 'QK': QK.mean().item(),
            'EVOU': self.EVOU.mean().item(),
            'PVOU': self.PVOU.mean().item(),
            # 'OV': OV.mean().item(),
            'EU': self.EU.mean().item(),
            # 'direct_path': direct_path.mean().item(),
            # 'M': M.mean().item(),
            'E_bar': E_bar.mean().item(),
            'P_bar': P_bar.mean().item(),
            'P_q': P_q.mean().item(),
            'E_q': E_q.mean().item(),
            'P_hat': P_hat.mean().item(),
            'P': self.P.mean().item(),
            'E': self.E.mean().item()
        }

        # Sort the means for consistent plotting
        sorted_means = sorted(tensor_means.items(), key=lambda x: x[1])

        fig, ax = plt.subplots(figsize=(10, 6))
        
        y_pos = range(len(sorted_means))
        values = [mean for _, mean in sorted_means]
        labels = [name for name, _ in sorted_means]

        ax.plot(values, y_pos, 'bo-')
        ax.set_yticks(y_pos)
        ax.set_yticklabels(labels)
        ax.set_xlabel('Mean Value')
        ax.set_title('Mean Values of Tensors')

        plt.tight_layout()
        plt.show()

        # l_max = torch.argmax(final_logits)
        return M


if __name__ == "__main__":
    # Generate random sequences
    num_sequences = 384000

    sequences = torch.randint(0, d_vocab, (num_sequences, n_ctx))

    dataset = TensorDataset(sequences)
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)

    model_seed = 613947648
    torch.manual_seed(model_seed)
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
            loss = F.cross_entropy(logits, targets)

            # Backward pass and optimization
            loss.backward()
            optimizer.step()

            if step % 100 == 0:
                print(f"Epoch: {epoch+1}, Step: {step}, Loss: {loss.item():.4f}")

    print("Training completed.")
    
    # Appendix G.2.3: SVD
    # breakpoint()
    U, S, V = torch.svd(model.EQKE)
    sigma_1 = torch.diag(S)[0]
    sigma_2 = torch.diag(S)[1]
    print("EQKE")
    print(f"Ratio of rank 1 and 2 singular values: {sigma_1/sigma_2}")

    # Save the trained model
    torch.save(model.state_dict(), "trained_transformer.pth")

    # Appendix B.2: Heatmaps
    # Create a list to store the heatmap images
    heatmap_images = []

    # Generate heatmaps and store them in the list
    heatmap_images.append(plot_heatmap(model.EQKE, "EQKE (Position-Independent Attention)", "key token", "query token"))
    heatmap_images.append(plot_heatmap(model.EQKP, "EQKP (Position-Dependent Attention)", "key position", "query token"))
    heatmap_images.append(plot_heatmap(model.EVOU, "EVOU (Value Output)", "output logit token", "input token"))
    heatmap_images.append(plot_heatmap(model.PVOU, "PVOU (Position-Dependent Output)", "output logit token", "input position"))
    heatmap_images.append(plot_heatmap(model.EU, "EU (Direct Path)", "output logit token", "input token"))

    # Create a grid of images
    num_cols = 2
    num_rows = (len(heatmap_images) + 1) // 2  # Round up to ensure all images fit
    fig, axes = plt.subplots(num_rows, num_cols, figsize=(20, 10 * num_rows))

    # Flatten the axes array for easier indexing
    axes = axes.flatten()

    # Add each heatmap to the grid
    for i, heatmap_fig in enumerate(heatmap_images):
        heatmap_fig.canvas.draw()     
        axes[i].imshow(heatmap_fig.canvas.buffer_rgba())
        axes[i].axis('off')

    # Remove any unused subplots
    for i in range(len(heatmap_images), len(axes)):
        fig.delaxes(axes[i])

    # Adjust the layout and save the figure
    plt.tight_layout()
    plt.savefig("heatmap_grid.png")
    plt.close(fig)