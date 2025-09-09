# train_reinforce.py
import torch, time
import torch.optim as optim
import numpy as np
from pointer_net import PointerNet
from data_utils import generate_euclidean_instances
from baselines import compute_length

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def batch_tour_length(coords_np, tours_np):
    # coords_np shape (B, n, 2), tours_np shape (B, n) int indices
    B, n, _ = coords_np.shape
    lens = []
    for i in range(B):
        c = coords_np[i]
        tour = tours_np[i].tolist()
        lens.append(compute_length(c, tour))
    return np.array(lens, dtype=np.float32)

def train():
    hidden = 128
    model = PointerNet(input_dim=2, hidden_dim=hidden).to(device)
    opt = optim.Adam(model.parameters(), lr=1e-4)
    epochs = 10000
    batch_size = 128
    n_nodes = 10
    baseline = None
    beta = 0.9  # baseline smoothing

    for it in range(1, epochs+1):
        coords = generate_euclidean_instances(batch_size, n_nodes)  # (B,n,2)
        coords_t = torch.from_numpy(coords).to(device)
        tours_pred, log_probs = model(coords_t, deterministic=False)  # (B,n)
        log_prob_sum = log_probs.sum(dim=1)  # (B,)
        tours_np = tours_pred.detach().cpu().numpy()
        lengths = batch_tour_length(coords, tours_np)  # (B,)
        reward = -lengths  # we want to minimize length -> maximize negative length

        if baseline is None:
            baseline = reward.mean()
        else:
            baseline = beta * baseline + (1 - beta) * reward.mean()

        advantage = reward - baseline
        loss = - (torch.from_numpy(advantage).to(device) * log_prob_sum).mean()
        opt.zero_grad()
        loss.backward()
        opt.step()

        if it % 100 == 0:
            print(f"Iter {it}, loss {loss.item():.4f}, avg_len {(-reward).mean():.4f}")

        if it % 1000 == 0:
            # evaluate small validation
            eval_coords = generate_euclidean_instances(128, n_nodes, seed=it)
            with torch.no_grad():
                tours_eval, _ = model(torch.from_numpy(eval_coords).to(device), deterministic=True)
                lens_eval = batch_tour_length(eval_coords, tours_eval.cpu().numpy())
                print(f"Eval: mean length {lens_eval.mean():.4f}")

    torch.save(model.state_dict(), "pointer_reinforce.pth")

if __name__ == "__main__":
    train()
