import pennylane as qml
import pennylane.numpy as np

class Ansatz:
    def __init__(self, wires):
        self.wires = wires

    def layer(self, x, params, i0=0, inc=1):
        """Building block of the embedding ansatz"""
        i = i0
        for j, wire in enumerate(self.wires):
            qml.Hadamard(wires=wire)
            qml.RZ(x[i % len(x)], wires=wire)
            i += inc
            qml.RY(params[0, j], wires=wire)

        for j in range(len(self.wires) - 1):
            qml.CRZ(params[1, j], wires=[self.wires[j], self.wires[j + 1]])

    def feature_map(self, x, params):
        """Applies the quantum feature map"""
        for j, layer_params in enumerate(params):
            self.layer(x, layer_params, i0=j * len(self.wires))
        qml.Barrier(wires=self.wires)  # Apply barrier to all wires



class Ansatz2:
    def __init__(self, wires):
        self.wires = wires

    def ansatz(self, x, params, wires):
        for layer_params in params:
            for idx, wire in enumerate(wires):
                qml.RX(layer_params[idx][0], wires=[wire])
                qml.RY(layer_params[idx][1], wires=[wire])
            for wire in [(wires[i], wires[i+1]) for i in range(0, len(wires) - 1, 2)]:
                qml.CZ(wires=[wire[0], wire[1]])
            for wire in [(wires[i], wires[i+1]) for i in range(1, len(wires) - 1, 2)]:
                qml.CZ(wires=[wire[0], wire[1]])
        qml.templates.embeddings.AngleEmbedding(
            features=x, wires=wires, rotation="X")
        qml.Barrier(wires=self.wires)  # Apply barrier to all wires


class ImprovedAnsatz:
    def __init__(self, wires):
        self.wires = wires
        
    def layer(self, x, params, layer_idx):
        """More expressive ansatz layer"""
        # Feature embedding
        for i, wire in enumerate(self.wires):
            qml.Hadamard(wires=wire)
            qml.RZ(x[i % len(x)] * (layer_idx+1), wires=wire)  # Layer-dependent scaling
            
        # Variational part
        for i, wire in enumerate(self.wires):
            qml.RY(params[0, i], wires=wire)
            qml.RZ(params[1, i], wires=wire)
            
        # Entanglement
        for i in range(len(self.wires)-1):
            qml.CNOT(wires=[self.wires[i], self.wires[i+1]])
            qml.CRY(params[2, i], wires=[self.wires[i], self.wires[i+1]])
            
    def feature_map(self, x, params):
        """Multi-layer feature map"""
        for j, layer_params in enumerate(params):
            self.layer(x, layer_params, j)


class ZZFeatureMap:
    def __init__(self, wires):
        self.wires = wires
        self.num_wires = len(wires)
        
    def layer(self, x, layer_idx):
        """
        Single layer of the ZZ feature map
        Args:
            x: Input data point
            params: Variational parameters (if any)
            layer_idx: Index of current layer
        """
        # First part: feature embedding
        for i, wire in enumerate(self.wires):
            qml.Hadamard(wires=wire)
            qml.RZ(2 * x[i % len(x)], wires=wire)
            
        # Second part: entangling gates
        for i in range(self.num_wires):
            for j in range(i+1, self.num_wires):
                # ZZ interaction between qubits i and j
                qml.CNOT(wires=[self.wires[i], self.wires[j]])
                qml.RZ(2 * (np.pi - x[i % len(x)]) * (np.pi - x[j % len(x)]), wires=self.wires[j])
                qml.CNOT(wires=[self.wires[i], self.wires[j]])

        def feature_map(self, x, num_layers):
            
            for layer_idx in range(num_layers):
                self.layer(x, params, layer_idx)
                qml.Barrier(wires=self.wires)  # Visual separation
            
            # Final Hadamard layer
            for wire in self.wires:
                qml.Hadamard(wires=wire)