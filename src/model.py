import numpy as np

def sigmoid(z: np.ndarray) -> np.ndarray:
    """
    Calculates the sigmoid function to map values to the range [0, 1].
    Uses np.clip to prevent overflow warnings for extreme values.

    Args:
        z (np.ndarray): The dot product of features and weights.

    Returns:
        np.ndarray: Probability that the input belongs to the positive class.
    """
    z = np.clip(z, -500, 500)
    return 1 / (1 + np.exp(-z))

def compute_cost(y: np.ndarray, h: np.ndarray, m: int, epsilon: float = 1e-15) -> float:
    """
    Calculates the binary cross-entropy loss (cost function).

    Args:
        y (np.ndarray): True labels (0 or 1).
        h (np.ndarray): Model predictions (probabilities).
        m (int): Number of training examples.
        epsilon (float): Threshold value to avoid log(0) error.

    Returns:
        float: The average error (loss).
    """
    h = np.clip(h, epsilon, 1 - epsilon)
    cost = -1/m * (np.dot(y.transpose(), np.log(h)) + np.dot((1-y).transpose(), np.log(1-h)))
    return float(cost)

def compute_gradient(x: np.ndarray, y: np.ndarray, h: np.ndarray, m: int) -> np.ndarray:
    """
    Calculates the gradient of the cost function for gradient descent.

    Args:
        x (np.ndarray): Feature matrix.
        y (np.ndarray): True labels.
        h (np.ndarray): Current predictions.
        m (int): Number of training examples.

    Returns:
        np.ndarray: The vector of gradients for each weight.
    """
    return 1/m * np.dot(x.transpose(), (h - y))

def update_weights(weight: np.ndarray, grad: np.ndarray, eta: float) -> np.ndarray:
    """
    Updates the weights in the direction opposite to the gradient.

    Args:
        weight (np.ndarray): Current weights.
        grad (np.ndarray): Calculated gradient.
        eta (float): Learning rate (alpha).

    Returns:
        np.ndarray: The new, updated weights.
    """
    return weight - (eta * grad)

def gradient_descent(x: np.ndarray, y: np.ndarray, alpha: float, num_iters: int) -> tuple[np.ndarray, list[float]]:
    """
    Trains the logistic regression model using gradient descent.

    Args:
        x (np.ndarray): Training features.
        y (np.ndarray): Training labels.
        alpha (float): The learning rate.
        num_iters (int): Number of training iterations (epochs).

    Returns:
        tuple: (The final trained weights, history of loss values).
    """
    m = x.shape[0]
    cost_history = []
    weight = np.zeros(x.shape[1])
    
    for i in range(num_iters):
        z = np.dot(x, weight)
        h = sigmoid(z)
        grad = compute_gradient(x, y, h, m)
        weight = update_weights(weight, grad, alpha)
        cost = compute_cost(y, h, m)
        cost_history.append(cost)
        
        if i % 100 == 0:
            print(f"Epoch {i:4d} | Loss: {cost:.4f}")
            
    return weight, cost_history

def predict(x: np.ndarray, weight: np.ndarray) -> np.ndarray:
    """
    Classifies new data points using the trained weights.

    Args:
        x (np.ndarray): Test features.
        weight (np.ndarray): Trained model weights.

    Returns:
        np.ndarray: Array of Booleans (True for positive, False for negative).
    """
    return sigmoid(np.dot(x, weight)) >= 0.5