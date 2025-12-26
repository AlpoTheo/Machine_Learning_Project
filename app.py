"""
CENG465 Machine Learning Project - Classification Toolkit

Authors:
    - Melek Arslan (220446018)
    - Alp Doruk Şengün (230444401)
    - Mehmet Anıl Ülkü (220441001)
    - Yusuf Özoğul (230446401)
    - Saip Deniz İnal (210444087)

How to Run:
    1. Install dependencies: pip install -r requirements.txt
    2. Run: python -m streamlit run app.py --server.headless true
    3. Open browser: http://localhost:8501
    4. Upload CSV, configure settings, train model

Features:
    - Perceptron classifier
    - Multilayer Perceptron with custom Backpropagation implementation
    - Decision Tree classifier
    - Preprocessing: Normalization (StandardScaler/MinMaxScaler), One-Hot Encoding
    - Metrics: Accuracy, Precision, Recall, F1-Score, Confusion Matrix
"""

import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, MinMaxScaler, LabelEncoder
from sklearn.linear_model import Perceptron
from sklearn.tree import DecisionTreeClassifier, plot_tree
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix, classification_report


# ============================================================================
# CUSTOM MULTILAYER NEURAL NETWORK WITH BACKPROPAGATION
# ============================================================================

class MultilayerNeuralNetwork:
    """
    Custom implementation of a Multilayer Perceptron with Backpropagation.
    This implementation uses numpy for matrix operations and supports
    configurable hidden layer architecture.
    """
    
    def __init__(self, hidden_layer_sizes=(100,), learning_rate=0.01, 
                 max_iter=1000, activation='sigmoid', random_state=None):
        """
        Initialize the neural network.
        
        Parameters:
        -----------
        hidden_layer_sizes : tuple
            Number of neurons in each hidden layer. e.g., (64, 32) means
            first hidden layer has 64 neurons, second has 32.
        learning_rate : float
            Step size for gradient descent weight updates.
        max_iter : int
            Maximum number of training epochs.
        activation : str
            Activation function: 'sigmoid' or 'relu'
        random_state : int
            Seed for reproducibility.
        """
        self.hidden_layer_sizes = hidden_layer_sizes
        self.learning_rate = learning_rate
        self.max_iter = max_iter
        self.activation = activation
        self.random_state = random_state
        
        # These will be initialized during fit()
        self.weights = []
        self.biases = []
        self.loss_history = []
        self.classes_ = None
        self.n_classes_ = None
        
    def _sigmoid(self, z):
        """Sigmoid activation function: σ(z) = 1 / (1 + e^(-z))"""
        # Clip to prevent overflow
        z = np.clip(z, -500, 500)
        return 1.0 / (1.0 + np.exp(-z))
    
    def _sigmoid_derivative(self, a):
        """Derivative of sigmoid: σ'(z) = σ(z) * (1 - σ(z))
        Note: 'a' is already the activated value σ(z)
        """
        return a * (1 - a)
    
    def _relu(self, z):
        """ReLU activation function: max(0, z)"""
        return np.maximum(0, z)
    
    def _relu_derivative(self, z):
        """Derivative of ReLU: 1 if z > 0, else 0"""
        return (z > 0).astype(float)
    
    def _activate(self, z, derivative=False):
        """Apply activation function or its derivative."""
        if self.activation == 'sigmoid':
            if derivative:
                return self._sigmoid_derivative(z)
            return self._sigmoid(z)
        elif self.activation == 'relu':
            if derivative:
                return self._relu_derivative(z)
            return self._relu(z)
        else:
            raise ValueError(f"Unknown activation: {self.activation}")
    
    def _softmax(self, z):
        """Softmax for output layer (multi-class classification)"""
        # Subtract max for numerical stability
        exp_z = np.exp(z - np.max(z, axis=1, keepdims=True))
        return exp_z / np.sum(exp_z, axis=1, keepdims=True)
    
    def _initialize_weights(self, n_features, n_outputs):
        """
        Initialize weights using Xavier/Glorot initialization.
        This helps prevent vanishing/exploding gradients.
        """
        if self.random_state is not None:
            np.random.seed(self.random_state)
        
        self.weights = []
        self.biases = []
        
        # Build the layer sizes: [input] + hidden_layers + [output]
        layer_sizes = [n_features] + list(self.hidden_layer_sizes) + [n_outputs]
        
        for i in range(len(layer_sizes) - 1):
            # Xavier initialization: scale by sqrt(2 / (fan_in + fan_out))
            limit = np.sqrt(6.0 / (layer_sizes[i] + layer_sizes[i+1]))
            W = np.random.uniform(-limit, limit, (layer_sizes[i], layer_sizes[i+1]))
            b = np.zeros((1, layer_sizes[i+1]))
            
            self.weights.append(W)
            self.biases.append(b)
    
    def _forward_propagation(self, X):
        """
        Forward pass through the network.
        
        Returns:
        --------
        activations : list
            Activated outputs at each layer (including input)
        z_values : list
            Pre-activation values (z = W*a + b) for each layer
        """
        activations = [X]
        z_values = []
        
        # Pass through hidden layers
        a = X
        for i in range(len(self.weights) - 1):
            z = np.dot(a, self.weights[i]) + self.biases[i]
            z_values.append(z)
            a = self._activate(z)
            activations.append(a)
        
        # Output layer (softmax for multi-class, sigmoid for binary)
        z_out = np.dot(a, self.weights[-1]) + self.biases[-1]
        z_values.append(z_out)
        
        if self.n_classes_ == 2:
            a_out = self._sigmoid(z_out)
        else:
            a_out = self._softmax(z_out)
        
        activations.append(a_out)
        
        return activations, z_values
    
    def _backward_propagation(self, X, y, activations, z_values):
        """
        Backward pass: compute gradients using chain rule.
        
        The core idea of backpropagation:
        - Error at output: δ_L = (a_L - y)
        - Error at hidden layers: δ_l = (W_{l+1}^T · δ_{l+1}) ⊙ σ'(z_l)
        - Gradient for weights: ∂L/∂W_l = a_{l-1}^T · δ_l
        - Gradient for biases: ∂L/∂b_l = sum(δ_l)
        """
        m = X.shape[0]  # Number of samples
        
        # Convert y to one-hot if multi-class
        if self.n_classes_ == 2:
            y_onehot = y.reshape(-1, 1)
        else:
            y_onehot = np.zeros((m, self.n_classes_))
            y_onehot[np.arange(m), y.astype(int)] = 1
        
        # Gradients storage
        dW = [None] * len(self.weights)
        db = [None] * len(self.biases)
        
        # Output layer error (delta)
        # For cross-entropy + softmax/sigmoid: δ = a - y
        delta = activations[-1] - y_onehot
        
        # Backpropagate through layers
        for i in range(len(self.weights) - 1, -1, -1):
            # Gradient for weights: dL/dW = a_prev^T · delta
            dW[i] = np.dot(activations[i].T, delta) / m
            
            # Gradient for biases: dL/db = mean(delta)
            db[i] = np.mean(delta, axis=0, keepdims=True)
            
            # Propagate error to previous layer (if not at input layer)
            if i > 0:
                # delta_prev = delta · W^T ⊙ activation_derivative(z_prev)
                delta = np.dot(delta, self.weights[i].T)
                delta = delta * self._activate(activations[i], derivative=True)
        
        return dW, db
    
    def _compute_loss(self, y_true, y_pred):
        """
        Compute cross-entropy loss.
        L = -1/m * Σ[y*log(p) + (1-y)*log(1-p)]
        """
        m = y_true.shape[0]
        epsilon = 1e-15  # Prevent log(0)
        y_pred = np.clip(y_pred, epsilon, 1 - epsilon)
        
        if self.n_classes_ == 2:
            y_true = y_true.reshape(-1, 1)
            loss = -np.mean(y_true * np.log(y_pred) + (1 - y_true) * np.log(1 - y_pred))
        else:
            y_onehot = np.zeros((m, self.n_classes_))
            y_onehot[np.arange(m), y_true.astype(int)] = 1
            loss = -np.mean(np.sum(y_onehot * np.log(y_pred), axis=1))
        
        return loss
    
    def fit(self, X, y):
        """
        Train the neural network using backpropagation.
        
        Parameters:
        -----------
        X : array-like, shape (n_samples, n_features)
            Training data
        y : array-like, shape (n_samples,)
            Target labels
        """
        X = np.array(X, dtype=np.float64)
        y = np.array(y, dtype=np.float64)
        
        # Determine number of classes
        self.classes_ = np.unique(y)
        self.n_classes_ = len(self.classes_)
        
        n_features = X.shape[1]
        n_outputs = 1 if self.n_classes_ == 2 else self.n_classes_
        
        # Initialize weights
        self._initialize_weights(n_features, n_outputs)
        
        self.loss_history = []
        
        # Training loop (Gradient Descent)
        for epoch in range(self.max_iter):
            # Forward pass
            activations, z_values = self._forward_propagation(X)
            
            # Compute loss
            loss = self._compute_loss(y, activations[-1])
            self.loss_history.append(loss)
            
            # Backward pass (compute gradients)
            dW, db = self._backward_propagation(X, y, activations, z_values)
            
            # Update weights and biases using gradient descent
            # W = W - learning_rate * dL/dW
            for i in range(len(self.weights)):
                self.weights[i] -= self.learning_rate * dW[i]
                self.biases[i] -= self.learning_rate * db[i]
        
        return self
    
    def predict_proba(self, X):
        """Predict class probabilities."""
        X = np.array(X, dtype=np.float64)
        activations, _ = self._forward_propagation(X)
        return activations[-1]
    
    def predict(self, X):
        """Predict class labels."""
        proba = self.predict_proba(X)
        
        if self.n_classes_ == 2:
            return (proba >= 0.5).astype(int).ravel()
        else:
            return np.argmax(proba, axis=1)
    
    def get_params(self, deep=True):
        """Get model parameters (sklearn compatibility)."""
        return {
            'hidden_layer_sizes': self.hidden_layer_sizes,
            'learning_rate': self.learning_rate,
            'max_iter': self.max_iter,
            'activation': self.activation,
            'random_state': self.random_state
        }


# ============================================================================
# STREAMLIT APPLICATION
# ============================================================================

# Page Configuration
st.set_page_config(page_title="CENG465 ML Toolkit", layout="wide", page_icon="🤖")

st.title("🎓 CENG465 - Group ML Toolkit")
st.markdown("This tool allows you to Train, Test and Evaluate ML models on any CSV dataset.")

# --- LEFT PANEL (Settings) ---
st.sidebar.header("1. Upload & Settings")
uploaded = st.sidebar.file_uploader("Upload your Dataset (CSV)", type=["csv"])

if uploaded:
    try:
        df = pd.read_csv(uploaded)
        
        # --- MAIN AREA (Data Preview) ---
        st.subheader("📊 Dataset Preview")
        st.dataframe(df.head())

        # --- SETTINGS ---
        st.sidebar.subheader("2. Preprocessing")
        
        # Target Column Selection
        cols = df.columns.tolist()
        target = st.sidebar.selectbox("Select Target Column (Class Label)", cols)

        # Preprocessing Options
        scaler_choice = st.sidebar.selectbox("Normalization Method", ["None", "StandardScaler", "MinMaxScaler"])
        encoding_choice = st.sidebar.checkbox("Apply One-Hot Encoding (Auto-detect categorical)", value=True)
        
        # Train/Test Split
        st.sidebar.subheader("3. Split & Model")
        test_size = st.sidebar.slider("Test Set Ratio", 0.1, 0.5, 0.3)
        
        # Model Selection
        model_name = st.sidebar.selectbox("Select Classifier", 
                                          ["Perceptron", 
                                           "Multilayer Perceptron (Backprop)", 
                                           "Decision Tree"])

        # --- MULTILAYER PERCEPTRON CONFIGURATION ---
        if model_name == "Multilayer Perceptron (Backprop)":
            st.sidebar.subheader("4. Multilayer Network Config")
            
            # Number of Hidden Layers
            n_hidden_layers = st.sidebar.slider(
                "Number of Hidden Layers", 
                min_value=1, 
                max_value=5, 
                value=2,
                help="How many hidden layers in the network"
            )
            
            # Neurons per layer - collect for each hidden layer
            hidden_layer_neurons = []
            st.sidebar.write("**Neurons per Hidden Layer:**")
            for i in range(n_hidden_layers):
                neurons = st.sidebar.slider(
                    f"Layer {i+1} Neurons",
                    min_value=4,
                    max_value=256,
                    value=64 if i == 0 else 32,
                    step=4,
                    help=f"Number of neurons in hidden layer {i+1}"
                )
                hidden_layer_neurons.append(neurons)
            
            # Convert to tuple for network initialization
            hidden_layer_sizes = tuple(hidden_layer_neurons)
            
            # Learning Rate
            learning_rate = st.sidebar.slider(
                "Learning Rate",
                min_value=0.001,
                max_value=1.0,
                value=0.1,
                step=0.001,
                format="%.3f",
                help="Step size for gradient descent. Smaller = more stable but slower learning"
            )
            
            # Max Iterations (Epochs)
            max_iter = st.sidebar.slider(
                "Max Iterations (Epochs)",
                min_value=100,
                max_value=5000,
                value=1000,
                step=100,
                help="Number of training epochs"
            )
            
            # Activation Function
            activation = st.sidebar.selectbox(
                "Activation Function",
                ["sigmoid", "relu"],
                help="Activation function for hidden layers"
            )
            
            # Display current network architecture
            st.sidebar.info(f"🧠 Network Architecture: Input → {' → '.join(map(str, hidden_layer_sizes))} → Output")

        # --- TRAIN BUTTON ---
        if st.sidebar.button("🚀 Train Model"):
            
            # 1. Data Preparation
            X = df.drop(columns=[target])
            y = df[target]

            # Encode target if it's categorical (e.g., Pass/Fail -> 0/1)
            if y.dtype == 'object':
                le = LabelEncoder()
                y = le.fit_transform(y)
                st.info(f"Target column encoded: {le.classes_}")

            # One-Hot Encoding (Automatic - as per PDF requirements)
            if encoding_choice:
                # Find and encode categorical columns
                X = pd.get_dummies(X, drop_first=True)
                st.write(f"Dataset shape after encoding: {X.shape}")

            # Normalization (as per PDF requirements)
            if scaler_choice == "StandardScaler":
                scaler = StandardScaler()
                X = pd.DataFrame(scaler.fit_transform(X), columns=X.columns)
            elif scaler_choice == "MinMaxScaler":
                scaler = MinMaxScaler()
                X = pd.DataFrame(scaler.fit_transform(X), columns=X.columns)

            # Split
            X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_size, random_state=42)

            # 2. Model Training
            if model_name == "Perceptron":
                model = Perceptron()
            elif model_name == "Multilayer Perceptron (Backprop)":
                # Use our custom implementation with user-defined parameters
                model = MultilayerNeuralNetwork(
                    hidden_layer_sizes=hidden_layer_sizes,
                    learning_rate=learning_rate,
                    max_iter=max_iter,
                    activation=activation,
                    random_state=42
                )
            else:
                model = DecisionTreeClassifier(criterion='entropy', random_state=42)
            
            # Training with progress indicator for MLP
            if model_name == "Multilayer Perceptron (Backprop)":
                with st.spinner(f"Training Multilayer Network with {max_iter} epochs..."):
                    model.fit(X_train, y_train)
            else:
                model.fit(X_train, y_train)
            
            y_pred = model.predict(X_test)

            # --- RESULTS SCREEN (Organized with Tabs) ---
            st.divider()
            st.header(f"Results for: {model_name}")
            
            tab1, tab2, tab3 = st.tabs(["📈 Metrics", "🟦 Confusion Matrix", "🌳 Model Details"])

            with tab1:
                col1, col2, col3, col4 = st.columns(4)
                col1.metric("Accuracy", f"{accuracy_score(y_test, y_pred):.4f}")
                col2.metric("Precision", f"{precision_score(y_test, y_pred, average='weighted', zero_division=0):.4f}")
                col3.metric("Recall", f"{recall_score(y_test, y_pred, average='weighted', zero_division=0):.4f}")
                col4.metric("F1 Score", f"{f1_score(y_test, y_pred, average='weighted', zero_division=0):.4f}")
                
                st.text("Detailed Classification Report:")
                st.code(classification_report(y_test, y_pred, zero_division=0))

            with tab2:
                st.write("Confusion Matrix Heatmap")
                cm = confusion_matrix(y_test, y_pred)
                fig, ax = plt.subplots(figsize=(6, 4))
                sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", ax=ax)
                st.pyplot(fig)

            with tab3:
                st.write("Model Parameters:")
                st.json(model.get_params())
                
                # Show training loss curve for MLP
                if model_name == "Multilayer Perceptron (Backprop)":
                    st.write("### Training Loss Curve (Backpropagation)")
                    st.write("This curve shows how the cross-entropy loss decreases during training.")
                    
                    fig_loss, ax_loss = plt.subplots(figsize=(10, 4))
                    ax_loss.plot(model.loss_history, color='blue', linewidth=1.5)
                    ax_loss.set_xlabel('Epoch')
                    ax_loss.set_ylabel('Cross-Entropy Loss')
                    ax_loss.set_title('Training Loss over Epochs')
                    ax_loss.grid(True, alpha=0.3)
                    st.pyplot(fig_loss)
                    
                    # Show final loss value
                    st.write(f"**Initial Loss:** {model.loss_history[0]:.4f}")
                    st.write(f"**Final Loss:** {model.loss_history[-1]:.4f}")
                    st.write(f"**Loss Reduction:** {((model.loss_history[0] - model.loss_history[-1]) / model.loss_history[0] * 100):.2f}%")
                    
                    # Network architecture visualization
                    st.write("### Network Architecture")
                    arch_str = f"Input ({X_train.shape[1]} features)"
                    for i, neurons in enumerate(hidden_layer_sizes):
                        arch_str += f" → Hidden Layer {i+1} ({neurons} neurons)"
                    n_outputs = len(np.unique(y_train))
                    arch_str += f" → Output ({n_outputs} classes)"
                    st.info(arch_str)
                
                # Decision Tree Visualization
                if model_name == "Decision Tree":
                    st.write("Decision Tree Visualization:")
                    fig_tree, ax_tree = plt.subplots(figsize=(20, 10))
                    plot_tree(model, filled=True, feature_names=X.columns, ax=ax_tree, max_depth=3)
                    st.pyplot(fig_tree)

    except Exception as e:
        st.error(f"An error occurred: {e}")
        st.warning("Tip: Please check your CSV file format.")

else:
    st.info("Please upload a CSV file from the left panel.")
