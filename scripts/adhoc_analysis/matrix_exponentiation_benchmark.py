import torch
import time
import matplotlib.pyplot as plt
from torch import nn

class ExpModel(nn.Module):
    def __init__(self, input_size):
        super().__init__()
        self.linear = nn.Linear(input_size, input_size, bias=False)
        
    def forward(self, x):
        A = self.linear.weight
        return x @ torch.matrix_exp(A)

class SimpleModel(nn.Module):
    def __init__(self, input_size):
        super().__init__()
        self.linear = nn.Linear(input_size, input_size, bias=False)
        
    def forward(self, x):
        return x @ self.linear.weight

def time_model(model, device, dim=64):
    x = torch.randn(dim, dim).to(device)
    target = torch.randn(dim, dim).to(device)
    
    # Warm-up
    for _ in range(3):
        out = model(x)
        loss = torch.norm(out - target)
        loss.backward()
    
    # Forward timing
    start = time.time()
    out = model(x)
    if device == 'cuda': torch.cuda.synchronize()
    forward_time = time.time() - start
    
    # Backward timing
    loss = torch.norm(out - target)
    start = time.time()
    loss.backward()
    if device == 'cuda': torch.cuda.synchronize()
    backward_time = time.time() - start
    
    return forward_time, backward_time

def compare_models(sizes, device='cpu', trials=10):
    results = []
    for size in sizes:
        exp_model = ExpModel(size).to(device)
        simple_model = SimpleModel(size).to(device)
        
        exp_fwd, exp_bwd = 0.0, 0.0
        simple_fwd, simple_bwd = 0.0, 0.0
        
        for _ in range(trials):
            f, b = time_model(exp_model, device, size)
            exp_fwd += f
            exp_bwd += b
            
            f, b = time_model(simple_model, device, size)
            simple_fwd += f
            simple_bwd += b
        
        results.append((
            size,
            exp_fwd/trials, exp_bwd/trials,
            simple_fwd/trials, simple_bwd/trials
        ))
    return results

def plot_results(results):
    sizes = [r[0] for r in results]
    exp_fwd = [r[1] for r in results]
    exp_bwd = [r[2] for r in results]
    simple_fwd = [r[3] for r in results]
    simple_bwd = [r[4] for r in results]
    
    # Compute compute factors
    forward_ratio = [e / s for e, s in zip(exp_fwd, simple_fwd)]
    backward_ratio = [e / s for e, s in zip(exp_bwd, simple_bwd)]
    
    fig, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=(18, 5))
    
    # Forward pass comparison
    ax1.plot(sizes, exp_fwd, label='With expm')
    ax1.plot(sizes, simple_fwd, label='Without expm')
    ax1.set_xlabel('Matrix Size')
    ax1.set_ylabel('Time (s)')
    ax1.set_title('Forward Pass Timing')
    ax1.legend()
    
    # Backward pass comparison
    ax2.plot(sizes, exp_bwd, label='With expm')
    ax2.plot(sizes, simple_bwd, label='Without expm')
    ax2.set_xlabel('Matrix Size')
    ax2.set_ylabel('Time (s)')
    ax2.set_title('Backward Pass Timing')
    
    # Compute factor plot
    ax3.plot(sizes, forward_ratio, marker='o', label='Forward')
    ax3.plot(sizes, backward_ratio, marker='o', label='Backward')
    ax3.set_xlabel('Matrix Size')
    ax3.set_ylabel('Time Ratio (expm/simple)')
    ax3.set_title('Compute Factor Comparison')
    ax3.legend()
    ax3.grid(True)
    
    plt.tight_layout()
    plt.show()

if __name__ == '__main__':
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    sizes = [16, 32, 64, 128, 256, 512, 1024, 2028]  # Adjust matrix sizes here
    results = compare_models(sizes, device=device, trials=500)
    plot_results(results)