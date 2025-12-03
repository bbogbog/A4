import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

# -------------- Part A ----------------- #

class SigmoidNeuralNetwork:
    def __init__(self, learning_rate=0.1):
        self.lr = learning_rate
        # Initialize weights and biases randomly
        # Weights between Input (2) and Hidden (2)
        np.random.seed(42)
        self.w1 = np.random.randn(2, 2)
        self.b1 = np.random.randn(1, 2)
        
        # Weights between Hidden (2) and Output (1)
        self.w2 = np.random.randn(2, 1)
        self.b2 = np.random.randn(1, 1)

    def sigmoid(self, s):
        return 1 / (1 + np.exp(-s))

    def sigmoid_derivative(self, s):        
        return s * (1 - s)

    def forward(self, X):
        # Input to Hidden
        self.z1 = np.dot(X, self.w1) + self.b1
        self.a1 = self.sigmoid(self.z1) # Hidden layer output (y1, y2)
        
        # Hidden to Output
        self.z2 = np.dot(self.a1, self.w2) + self.b2
        self.a2 = self.sigmoid(self.z2) # Final output
        return self.a2

    def backward(self, X, y, output):
        # Calculate Error
        self.output_error = y - output
        self.output_delta = self.output_error * self.sigmoid_derivative(output)

        self.hidden_error = self.output_delta.dot(self.w2.T)
        self.hidden_delta = self.hidden_error * self.sigmoid_derivative(self.a1)

        # Update Weights and Biases
        self.w2 += self.a1.T.dot(self.output_delta) * self.lr
        self.b2 += np.sum(self.output_delta, axis=0, keepdims=True) * self.lr
        self.w1 += X.T.dot(self.hidden_delta) * self.lr
        self.b1 += np.sum(self.hidden_delta, axis=0, keepdims=True) * self.lr

    def train(self, X, y, epochs=10000):
        for _ in range(epochs):
            output = self.forward(X)
            self.backward(X, y, output)


X = np.array([[-1, -1],
              [-1, 1],
              [1, -1],
              [1, 1]])


y = np.array([[0], [1], [1], [0]])

# Train the Model
nn = SigmoidNeuralNetwork(learning_rate=0.5)
nn.train(X, y, epochs=20000)


# -------------- Part B ----------------- #

print("--- Part (b): Trained Weights ---")
print("Input-to-Hidden Weights (w1):\n", nn.w1)
print("Hidden Bias (b1):\n", nn.b1)
print("Hidden-to-Output Weights (w2):\n", nn.w2)
print("Output Bias (b2):\n", nn.b2)

# Visualization Setup
def plot_surface(func, title, pos):
    fig = plt.figure(figsize=(15, 5))
    
    # Create meshgrid for x1, x2 space
    x_range = np.linspace(-2, 2, 50)
    y_range = np.linspace(-2, 2, 50)
    XX, YY = np.meshgrid(x_range, y_range)
    
    # Flatten to pass through network
    grid_input = np.c_[XX.ravel(), YY.ravel()]
    
    # Manual forward pass to capture different layers
    z1 = np.dot(grid_input, nn.w1) + nn.b1
    a1 = 1 / (1 + np.exp(-z1)) # Hidden outputs
    z2 = np.dot(a1, nn.w2) + nn.b2
    a2 = 1 / (1 + np.exp(-z2)) # Final output
    
    # Reshape for plotting
    Hidden1 = a1[:, 0].reshape(XX.shape)
    Hidden2 = a1[:, 1].reshape(XX.shape)
    Output = a2.reshape(XX.shape)
    
    # Plot Hidden Unit 1
    ax1 = fig.add_subplot(131, projection='3d')
    ax1.plot_surface(XX, YY, Hidden1, cmap='viridis')
    ax1.set_title('Hidden Unit 1 Activation')
    ax1.set_xlabel('x1')
    ax1.set_ylabel('x2')
    
    # Plot Hidden Unit 2
    ax2 = fig.add_subplot(132, projection='3d')
    ax2.plot_surface(XX, YY, Hidden2, cmap='viridis')
    ax2.set_title('Hidden Unit 2 Activation')
    ax2.set_xlabel('x1')
    ax2.set_ylabel('x2')

    # Plot Output Unit
    ax3 = fig.add_subplot(133, projection='3d')
    ax3.plot_surface(XX, YY, Output, cmap='magma')
    ax3.set_title('Final Output Activation')
    ax3.set_xlabel('x1')
    ax3.set_ylabel('x2')
    
    # plt.tight_layout()
    plt.savefig("Q3-Part_B.png")
    plt.show()

plot_surface(nn, "Activations", 1)

# --------------- Part C ----------------- #

plt.figure(figsize=(8, 6))

# Create a mesh for contour plot
xx, yy = np.meshgrid(np.linspace(-2, 2, 200), np.linspace(-2, 2, 200))
grid = np.c_[xx.ravel(), yy.ravel()]
preds = nn.forward(grid)
preds = preds.reshape(xx.shape)

# Plot decision boundary
plt.contourf(xx, yy, preds, levels=50, cmap="RdBu", alpha=0.6)
plt.colorbar(label='Network Output')

# Plot original training points
plt.scatter([-1, 1], [-1, 1], c='blue', s=100, edgecolors='k', label='Class 0 (Low)')

plt.scatter([-1, 1], [1, -1], c='red', s=100, marker='s', edgecolors='k', label='Class 1 (High)')

plt.title("Part (c): Decision Boundary in Input Space")
plt.xlabel("x1")
plt.ylabel("x2")
plt.legend()
plt.grid(True)
plt.savefig("Q3-Part_C.png")
plt.show()

# -------------- Part D ----------------- #

# Forward pass for x = [0, 0]
zero_input = np.array([[0, 0]])
# We need to manually access the hidden layer 'a1' for this input
z1_zero = np.dot(zero_input, nn.w1) + nn.b1
hidden_representation_zero = 1 / (1 + np.exp(-z1_zero))

# Get representations for training points too for context
z1_train = np.dot(X, nn.w1) + nn.b1
hidden_train = 1 / (1 + np.exp(-z1_train))

plt.figure(figsize=(8, 6))
plt.title("Part (d): Hidden Space Representation (y1, y2)")

# Plot training points in hidden space
plt.scatter(hidden_train[[0, 3], 0], hidden_train[[0, 3], 1], c='blue', s=100, label='Class 0')
plt.scatter(hidden_train[[1, 2], 0], hidden_train[[1, 2], 1], c='red', s=100, marker='s', label='Class 1')

# Plot the x=0 point
plt.scatter(hidden_representation_zero[0,0], hidden_representation_zero[0,1], 
            c='green', s=150, marker='*', edgecolors='k', label='x = [0,0]')

plt.xlabel("Hidden Unit 1 Activation (y1)")
plt.ylabel("Hidden Unit 2 Activation (y2)")
plt.legend()
plt.grid(True)
plt.savefig("Q3-Part_D.png")
plt.show()

print(f"Representation of x=[0,0] in hidden space: {hidden_representation_zero}")