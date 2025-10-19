import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import h5py
import json
from torch.utils.data import Dataset, DataLoader
import torch.optim as optim
from typing import Dict, List, Tuple, Optional
import logging

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


class EntanglementDataset(Dataset):
    def __init__(self, h5_file_path: str):
        self.h5_file_path = h5_file_path
        self.sample_indices = []

        with h5py.File(h5_file_path, 'r') as f:
            if 'proxy_features' in f:
                self.sample_indices = list(range(len(f['proxy_features'])))

    def __len__(self):
        return len(self.sample_indices)

    def __getitem__(self, idx):
        with h5py.File(self.h5_file_path, 'r') as f:
            
            features = f['proxy_features'][f'sample_{idx}'][()]

            
            params_str = f['proxy_features'][f'sample_{idx}'].attrs['hamiltonian_params']
            params = json.loads(params_str)

           
            hamiltonian_features = self._params_to_features(params)

            return {
                'proxy_features': torch.FloatTensor(features),
                'hamiltonian_features': torch.FloatTensor(hamiltonian_features),
                'model_type': params['model']
            }

    def _params_to_features(self, params: Dict) -> List[float]:
        
        features = []

        # ویژگی‌های پایه
        features.append(params.get('lattice_size', 4) / 10.0)  # نرمال‌سازی

        
        if params['model'] == 'TFIM':
            features.extend([
                params.get('J', 1.0),
                params.get('h', 1.0),
                1.0 if params.get('boundary') == 'periodic' else 0.0,
                0.0, 0.0, 0.0  
            ])
        elif params['model'] == 'Heisenberg':
            features.extend([
                params.get('J', 1.0),
                params.get('delta', 1.0),
                1.0 if params.get('boundary') == 'periodic' else 0.0,
                0.0, 0.0  # padding
            ])
        else:  # ToricCode
            features.extend([
                params.get('J_star', 1.0),
                params.get('J_plaquette', 1.0),
                1.0,  
                0.0  # padding
            ])

        
        while len(features) < 7:
            features.append(0.0)

        return features[:7]


class ArchitectureBlock(nn.Module):
  

    def __init__(self, block_type: str, hidden_dim: int = 64):
        super().__init__()
        self.block_type = block_type

        if block_type == 'RBM':
            self.layer = nn.Linear(hidden_dim, hidden_dim)
        elif block_type == 'MPS':
            self.layer = nn.Linear(hidden_dim, hidden_dim)
            self.attention = nn.MultiheadAttention(hidden_dim, num_heads=4)
        elif block_type == 'Attention':
            self.attention = nn.MultiheadAttention(hidden_dim, num_heads=8)
            self.norm = nn.LayerNorm(hidden_dim)
        elif block_type == 'FC':
            self.layer = nn.Sequential(
                nn.Linear(hidden_dim, hidden_dim * 2),
                nn.ReLU(),
                nn.Linear(hidden_dim * 2, hidden_dim)
            )
        else:
            raise ValueError(f"Unknown block type: {block_type}")

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if self.block_type == 'RBM':
            return torch.tanh(self.layer(x))
        elif self.block_type == 'MPS':
            x1 = self.layer(x)
            x2, _ = self.attention(x.unsqueeze(0), x.unsqueeze(0), x.unsqueeze(0))
            return torch.tanh(x1 + x2.squeeze(0))
        elif self.block_type == 'Attention':
            attn_out, _ = self.attention(x.unsqueeze(0), x.unsqueeze(0), x.unsqueeze(0))
            return self.norm(x + attn_out.squeeze(0))
        elif self.block_type == 'FC':
            return self.layer(x)

        return x


class DifferentiableSupernet(nn.Module):
    

    def __init__(self,
                 input_dim: int = 7,
                 hidden_dim: int = 64,
                 output_dim: int = 4,
                 num_blocks: int = 3):
        super().__init__()

        self.hidden_dim = hidden_dim
        self.num_blocks = num_blocks

    
        self.block_types = ['RBM', 'MPS', 'Attention', 'FC']

        
        self.input_proj = nn.Linear(input_dim, hidden_dim)

        
        self.blocks = nn.ModuleDict()
        for block_type in self.block_types:
            self.blocks[block_type] = ArchitectureBlock(block_type, hidden_dim)

        
        self.architecture_weights = nn.Parameter(
            torch.ones(len(self.block_types), num_blocks) / len(self.block_types)
        )

        
        self.output_layer = nn.Linear(hidden_dim, output_dim)

        
        self.dropout = nn.Dropout(0.1)

    def forward(self,
                hamiltonian_features: torch.Tensor,
                proxy_target: Optional[torch.Tensor] = None,
                temperature: float = 1.0) -> Tuple[torch.Tensor, torch.Tensor]:

        
        x = torch.relu(self.input_proj(hamiltonian_features))

        
        arch_probs = F.gumbel_softmax(self.architecture_weights, tau=temperature, dim=0)

        
        for block_idx in range(self.num_blocks):
            block_outputs = []

            for i, block_type in enumerate(self.block_types):
                block_out = self.blocks[block_type](x)
                weight = arch_probs[i, block_idx]
                block_outputs.append(weight * block_out)

        
            x = sum(block_outputs)
            x = self.dropout(x)

        
        output = self.output_layer(x)

        return output, arch_probs

    def get_optimal_architecture(self) -> List[str]:
        
        with torch.no_grad():
            arch_probs = F.softmax(self.architecture_weights, dim=0)
            chosen_blocks = []

            for block_idx in range(self.num_blocks):
                block_probs = arch_probs[:, block_idx]
                chosen_block_idx = torch.argmax(block_probs).item()
                chosen_blocks.append(self.block_types[chosen_block_idx])

            return chosen_blocks


class QED_NAS_Trainer:
    

    def __init__(self,
                 h5_file_path: str,
                 hidden_dim: int = 64,
                 num_blocks: int = 3,
                 learning_rate: float = 0.001):

        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        logger.info(f"Using device: {self.device}")

        
        self.dataset = EntanglementDataset(h5_file_path)
        self.dataloader = DataLoader(self.dataset, batch_size=16, shuffle=True)

    
        self.supernet = DifferentiableSupernet(
            hidden_dim=hidden_dim,
            num_blocks=num_blocks
        ).to(self.device)

        
        self.optimizer = optim.AdamW(
            self.supernet.parameters(),
            lr=learning_rate,
            weight_decay=0.01
        )

        
        self.initial_temperature = 1.0
        self.final_temperature = 0.1
        self.temperature_decay = 0.95
        self.current_temperature = self.initial_temperature

        
        self.history = {
            'total_loss': [],
            'task_loss': [],
            'entanglement_loss': [],
            'temperature': []
        }

    def combined_loss_function(self,
                               prediction: torch.Tensor,
                               target: torch.Tensor,
                               architecture_weights: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        

        
        task_loss = F.mse_loss(prediction, target)

        
        
        arch_probs = F.softmax(architecture_weights, dim=0)
        entropy_loss = -torch.sum(arch_probs * torch.log(arch_probs + 1e-8)) / architecture_weights.numel()
        entanglement_loss = -entropy_loss  # ماکزیمم کردن آنتروپی برای اکتشاف

        
        alpha = 0.7  
        beta = 0.3  

        total_loss = alpha * task_loss + beta * entanglement_loss

        return total_loss, task_loss, entanglement_loss

    def train_epoch(self) -> Dict[str, float]:
       
        self.supernet.train()
        total_loss = 0.0
        total_task_loss = 0.0
        total_entanglement_loss = 0.0
        num_batches = 0

        for batch in self.dataloader:
            self.optimizer.zero_grad()

       
            hamiltonian_features = batch['hamiltonian_features'].to(self.device)
            proxy_target = batch['proxy_features'].to(self.device)

            # forward pass
            prediction, arch_weights = self.supernet(
                hamiltonian_features,
                proxy_target,
                temperature=self.current_temperature
            )

       
            total_loss_batch, task_loss_batch, entanglement_loss_batch = \
                self.combined_loss_function(prediction, proxy_target, arch_weights)

            # backward pass
            total_loss_batch.backward()
            self.optimizer.step()

       
            total_loss += total_loss_batch.item()
            total_task_loss += task_loss_batch.item()
            total_entanglement_loss += entanglement_loss_batch.item()
            num_batches += 1

       
        self.current_temperature = max(
            self.final_temperature,
            self.current_temperature * self.temperature_decay
        )

        return {
            'total_loss': total_loss / num_batches,
            'task_loss': total_task_loss / num_batches,
            'entanglement_loss': total_entanglement_loss / num_batches,
            'temperature': self.current_temperature
        }

    def train(self, num_epochs: int = 100):
       
        logger.info("Starting QED-NAS training...")

        for epoch in range(num_epochs):
            metrics = self.train_epoch()

       
            for key, value in metrics.items():
                self.history[key].append(value)

            if (epoch + 1) % 10 == 0:
                logger.info(
                    f"Epoch {epoch + 1}/{num_epochs} | "
                    f"Total Loss: {metrics['total_loss']:.4f} | "
                    f"Task Loss: {metrics['task_loss']:.4f} | "
                    f"Entanglement Loss: {metrics['entanglement_loss']:.4f} | "
                    f"Temp: {metrics['temperature']:.3f}"
                )

       
        optimal_architecture = self.supernet.get_optimal_architecture()
        logger.info(f"Optimal architecture: {optimal_architecture}")

        return optimal_architecture

    def evaluate(self) -> Dict[str, float]:
       
        self.supernet.eval()
        total_mse = 0.0
        num_samples = 0

        with torch.no_grad():
            for batch in self.dataloader:
                hamiltonian_features = batch['hamiltonian_features'].to(self.device)
                proxy_target = batch['proxy_features'].to(self.device)

                prediction, _ = self.supernet(hamiltonian_features, temperature=0.1)
                mse = F.mse_loss(prediction, proxy_target)

                total_mse += mse.item() * len(batch['proxy_features'])
                num_samples += len(batch['proxy_features'])

        return {'mse': total_mse / num_samples}

    def save_model(self, filepath: str):
       
        torch.save({
            'supernet_state_dict': self.supernet.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'history': self.history,
            'optimal_architecture': self.supernet.get_optimal_architecture()
        }, filepath)
        logger.info(f"Model saved to {filepath}")


def analyze_architecture_performance(optimal_architecture: List[str],
                                     h5_file_path: str):
    
    logger.info("Analyzing architecture performance...")

    
    block_counts = {}
    for block in optimal_architecture:
        block_counts[block] = block_counts.get(block, 0) + 1

    print("\n" + "=" * 60)
    print("ARCHITECTURE ANALYSIS RESULTS")
    print("=" * 60)
    print(f"Optimal Architecture: {' -> '.join(optimal_architecture)}")
    print("\nBlock Distribution:")
    for block, count in block_counts.items():
        print(f"  {block}: {count} blocks ({count / len(optimal_architecture) * 100:.1f}%)")

    
    print("\nArchitecture Patterns:")
    patterns = {}
    for i in range(len(optimal_architecture) - 1):
        pattern = f"{optimal_architecture[i]}-{optimal_architecture[i + 1]}"
        patterns[pattern] = patterns.get(pattern, 0) + 1

    for pattern, count in patterns.items():
        print(f"  {pattern}: {count} occurrences")


def main():
    
    print("QED-NAS PHASE 2: ARCHITECTURE OPTIMIZATION WITH SUPERNET")
    print("=" * 70)

    
    h5_file_path = 'quantum_dataset.h5'

    
    trainer = QED_NAS_Trainer(
        h5_file_path=h5_file_path,
        hidden_dim=64,
        num_blocks=4,  
        learning_rate=0.001
    )

    
    optimal_architecture = trainer.train(num_epochs=100)

    
    evaluation_results = trainer.evaluate()
    print(f"\nFinal Evaluation - MSE: {evaluation_results['mse']:.6f}")

    
    trainer.save_model('qed_nas_supernet.pth')

    
    analyze_architecture_performance(optimal_architecture, h5_file_path)

    print("\n" + "=" * 70)
    print("PHASE 2 COMPLETED SUCCESSFULLY!")
    print("=" * 70)


if __name__ == "__main__":
    main()