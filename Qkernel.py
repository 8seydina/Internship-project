import pennylane as qml
import pennylane.numpy as np
from Ansatz import Ansatz, Ansatz2, ImprovedAnsatz, ZZFeatureMap

class QKernel:
    def __init__(self, num_wires, ansatz, num_layers=6, device_name="default.qubit", shots=None):
        self.num_wires = num_wires
        self.wires = list(range(num_wires))
        self.ansatz = ansatz(self.wires)
        self.dev = qml.device(device_name, wires=num_wires, shots=shots)
        self.params = self.random_params(num_layers)

        self.kernel_circuit = self._build_qnode()

    def random_params(self, num_layers):
        return np.random.uniform(
            0, 2 * np.pi, 
            (num_layers, 3, max(self.num_wires, self.num_wires-1)),
            requires_grad=True
        )

    def _build_qnode(self):
        @qml.qnode(self.dev)
        def circuit(x1, x2, params):
            # Apply the first ansatz with x1
            if isinstance(self.ansatz, Ansatz):
                self.ansatz.feature_map(x1, params)
                qml.adjoint(self.ansatz.feature_map)(x2, params)
            elif isinstance(self.ansatz, Ansatz2):
                self.ansatz.ansatz(x1, params, self.wires)
                qml.adjoint(self.ansatz.ansatz)(x2, params, self.wires)
            elif isinstance(self.ansatz, ImprovedAnsatz):
                self.ansatz.feature_map(x1, params)
                qml.adjoint(self.ansatz.feature_map)(x2, params)
            elif isinstance(self.ansatz, ZZFeatureMap):
                self.ansatz.feature_map(x1, num_layers)
                qml.adjoint(self.ansatz.feature_map)(x2, num_layers)
        
            return qml.probs(wires=self.wires)
        return circuit

    def compute(self, x1, x2):
        return self.kernel_circuit(x1, x2, self.params)[0]

    def visualize(self, x1, x2):
        # Run the circuit once to generate the Qiskit circuit
        _ = self.kernel_circuit(x1, x2, self.params)

        # Return the Qiskit circuit drawing
        return self.dev._circuit.draw(output='mpl')