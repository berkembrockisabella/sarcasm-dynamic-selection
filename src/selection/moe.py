import copy
import random
import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader, TensorDataset
from .moe_gate import GatingNetwork

class MixtureOfExperts:
    """
    Soft Mixture of Experts para combinação de uma pool
    de classificadores previamente treinados.

    Somente a gating network é treinada.
    """
    
    def __init__(self, input_dim, n_experts, hidden_dim=32, dropout=0.2, learning_rate=1e-3,
        weight_decay=1e-4, batch_size=32, max_epochs=200, patience=20, random_state=42, device=None):

        self.input_dim = input_dim
        self.n_experts = n_experts
        self.hidden_dim = hidden_dim
        self.dropout = dropout
        self.learning_rate = learning_rate
        self.weight_decay = weight_decay
        self.batch_size = batch_size
        self.max_epochs = max_epochs
        self.patience = patience
        self.random_state = random_state
        
        if device is None:
            self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        else:
            self.device = torch.device(device)
        
        self.gate = GatingNetwork(input_dim, n_experts, hidden_dim, dropout).to(self.device)
        
        self.history = {"train_loss": [], "val_loss": []} # verificar
        self.is_fitted = False
        
        self._set_seed()


    def _set_seed(self):
        random.seed(self.random_state)
        np.random.seed(self.random_state)
        torch.manual_seed(self.random_state)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(self.random_state)
    
    def _to_tensor(self, array):
        return torch.tensor(np.asarray(array), dtype=torch.float32)
    
    def fit(self, X_gate, expert_probabilities, y, X_val=None, expert_probabilities_val=None, y_val=None):
        """
        Treina a gating network
        
        X_gate -> features utilizadas pelo gate
        expert_probabilities -> probabilidades produzidas pelos especialistas
        y -> rótulos alvo, labels verdadeiras
        X_val -> features de validação para early stopping (opcional)
        """
        X_gate = np.asarray(X_gate)
        expert_probabilities = np.asarray(expert_probabilities)
        y = np.asarray(y)
        
        if X_gate.shape[0] != len(y) or expert_probabilities.shape[0] != len(y):
            raise ValueError("O número de amostras em X_gate, expert_probabilities e y deve ser o mesmo.")
        if expert_probabilities.shape[1] != self.n_experts:
            raise ValueError(f"O número de especialistas em expert_probabilities deve ser {self.n_experts}.")
        
        dataset = TensorDataset(self._to_tensor(X_gate), self._to_tensor(expert_probabilities), self._to_tensor(y))
        loader = DataLoader(dataset, batch_size=self.batch_size, shuffle=True)
        optimizer = torch.optim.Adam(self.gate.parameters(), lr=self.learning_rate, weight_decay=self.weight_decay)
        
        history = []
        best_loss = np.inf
        best_state = None
        epochs_no_improve = 0
        
        for epoch in range(1, self.max_epochs + 1):
            self.gate.train()
            epoch_losses = []
            
            for(gate_x, expert_probs, labels) in loader:
                gate_x = gate_x.to(self.device)
                expert_probs = expert_probs.to(self.device)
                labels = labels.to(self.device)
                
                optimizer.zero_grad()
                weights = self.gate(gate_x)
                
                # soft moe
                final_prob = torch.sum(weights * expert_probs, dim=1)
                final_prob = torch.clamp(final_prob, min=1e-7, max=1 - 1e-7)
                
                loss = F.binary_cross_entropy(final_prob, labels)
                loss.backward()
                optimizer.step()
                epoch_losses.append(loss.item())
            
            train_loss = np.mean(epoch_losses)
            
            # Validação
            if(X_val is not None and expert_probabilities_val is not None and y_val is not None):
                val_loss = self._loss(X_val, expert_probabilities_val, y_val)
                monitor_loss = val_loss
            else:
                val_loss = np.nan
                monitor_loss = train_loss
            
            history.append({
                "epoch": epoch,
                "train_loss": train_loss,
                "val_loss": val_loss
            })
            
            # early stopping
            if monitor_loss < best_loss:
                best_loss = monitor_loss
                best_state = copy.deepcopy(self.gate.state_dict())
                epochs_no_improve = 0
            else:
                epochs_no_improve += 1
            
            if(epochs_no_improve >= self.patience):
                print(f"Early stopping at epoch {epoch}. Best loss: {best_loss:.4f}")
                break
        
        # Recuperar melhor estado do gate
        if best_state is not None:
            self.gate.load_state_dict(best_state)
        self.history = history
        self.is_fitted = True
        
        return self
    
    
    def _loss(self, X_gate, expert_probabilities, y):
        probabilities = self.predict(X_gate, expert_probabilities, check_fitted=False)
        probabilities = np.clip(probabilities, 1e-7, 1 - 1e-7)
        
        y = np.asarray(y)
        loss = -np.mean(y * np.log(probabilities) + (1 - y) * np.log(1 - probabilities))
        return loss
    
    
    def get_gate_weights(self, X_gate):
        self.gate.eval()
        
        X = self._to_tensor(X_gate).to(self.device)
        with torch.no_grad():

            weights = self.gate(
                X
            )

        return (
            weights
            .cpu()
            .numpy()
        )


    # =====================================================
    # Predict proba
    # =====================================================

    def predict_proba(
        self,
        X_gate,
        expert_probabilities,
        check_fitted=True,
    ):

        if (
            check_fitted
            and not self.is_fitted_
        ):
            raise RuntimeError(
                "O MixtureOfExperts ainda "
                "não foi treinado."
            )

        expert_probabilities = np.asarray(
            expert_probabilities
        )

        weights = self.get_gate_weights(
            X_gate
        )

        if (
            weights.shape
            != expert_probabilities.shape
        ):
            raise ValueError(
                "Gate weights e expert probabilities "
                "possuem dimensões incompatíveis."
            )

        probabilities = np.sum(
            weights
            * expert_probabilities,
            axis=1,
        )

        return probabilities


    # =====================================================
    # Predict
    # =====================================================

    def predict(
        self,
        X_gate,
        expert_probabilities,
        threshold=0.5,
    ):

        probabilities = self.predict_proba(
            X_gate,
            expert_probabilities,
        )

        return (
            probabilities >= threshold
        ).astype(int)
    
    def forward(self, gate_features, expert_probabilities):
        weights = self.gate(gate_features)
        final_probability = torch.sum(weights * expert_probabilities, dim=1)
        return (final_probability, weights)