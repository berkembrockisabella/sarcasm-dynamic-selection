'''
Definir o gate utilizado pelo MoE
'''

import torch
import torch.nn as nn

class GatingNetwork(nn.Module):
    '''
    Gate Multi Layer Perceptron pequeno
    input -> dense(32) -> reLu -> dropout -> dense(7) -> softmax
    
    Para cada amostra, produz uma distribuição de pesos
    sobre os especialistas da pool.
    '''
    
    def __init__(self, input_dim, n_experts=7, hidden_dim=32, dropout=0.2,):
        super().__init__()

        self.network = nn.Sequential(
            nn.Linear(input_dim, hidden_dim,),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, n_experts)
        )

    def forward(self, x):
        logits = self.network(x)
        weights = torch.softmax(logits, dim=1,)

        return weights