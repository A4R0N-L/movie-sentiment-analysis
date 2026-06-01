import numpy as np

def sigmoid(z: np.ndarray) -> np.ndarray:
    """
    Berechnet die Sigmoid-Funktion, um Werte auf den Bereich [0, 1] abzubilden.
    Nutzt np.clip, um Overflow-Warnungen bei extremen Werten zu verhindern.

    Args:
        z (np.ndarray): Das Skalarprodukt aus Features und Gewichten.

    Returns:
        np.ndarray: Wahrscheinlichkeit, dass die Eingabe zur positiven Klasse gehört.
    """
    z = np.clip(z, -500, 500)
    return 1 / (1 + np.exp(-z))

def compute_cost(y: np.ndarray, h: np.ndarray, m: int, epsilon: float = 1e-15) -> float:
    """
    Berechnet die Binary Cross-Entropy Loss (Kostenfunktion).

    Args:
        y (np.ndarray): Wahre Labels (0 oder 1).
        h (np.ndarray): Vorhersagen des Modells (Wahrscheinlichkeiten).
        m (int): Anzahl der Trainingsbeispiele.
        epsilon (float): Minimalwert, um log(0) Fehler zu vermeiden.

    Returns:
        float: Der durchschnittliche Fehler (Loss).
    """
    h = np.clip(h, epsilon, 1 - epsilon)
    cost = -1/m * (np.dot(y.transpose(), np.log(h)) + np.dot((1-y).transpose(), np.log(1-h)))
    return float(cost)

def compute_gradient(x: np.ndarray, y: np.ndarray, h: np.ndarray, m: int) -> np.ndarray:
    """
    Berechnet den Gradienten der Kostenfunktion für den Gradientenabstieg.

    Args:
        x (np.ndarray): Merkmalsmatrix.
        y (np.ndarray): Wahre Labels.
        h (np.ndarray): Aktuelle Vorhersagen.
        m (int): Anzahl der Trainingsbeispiele.

    Returns:
        np.ndarray: Der Vektor der Gradienten für jedes Gewicht.
    """
    return 1/m * np.dot(x.transpose(), (h - y))

def update_weights(weight: np.ndarray, grad: np.ndarray, eta: float) -> np.ndarray:
    """
    Aktualisiert die Gewichte in entgegengesetzter Richtung des Gradienten.

    Args:
        weight (np.ndarray): Aktuelle Gewichte.
        grad (np.ndarray): Berechneter Gradient.
        eta (float): Lernrate (Alpha).

    Returns:
        np.ndarray: Die neuen, angepassten Gewichte.
    """
    return weight - (eta * grad)

def gradient_descent(x: np.ndarray, y: np.ndarray, alpha: float, num_iters: int) -> tuple[np.ndarray, list[float]]:
    """
    Trainiert das Logistic Regression Modell mittels Gradientenabstieg.

    Args:
        x (np.ndarray): Trainings-Features.
        y (np.ndarray): Trainings-Labels.
        alpha (float): Die Lernrate.
        num_iters (int): Anzahl der Trainings-Iterationen (Epochen).

    Returns:
        tuple: (Die final trainierten Gewichte, Historie der Loss-Werte).
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
    Klassifiziert neue Datenpunkte anhand der trainierten Gewichte.

    Args:
        x (np.ndarray): Test-Features.
        weight (np.ndarray): Trainierte Modellgewichte.

    Returns:
        np.ndarray: Array von Booleans (True für positiv, False für negativ).
    """
    return sigmoid(np.dot(x, weight)) >= 0.5