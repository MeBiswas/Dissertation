# src/fann_classifier/step_2.py

from sklearn.neural_network import MLPClassifier

# ═════════════════════════════════════════════════════════════════════════════
# PART 2 — FANN ARCHITECTURE
# ═════════════════════════════════════════════════════════════════════════════
 
def build_fann() -> MLPClassifier:
    model = MLPClassifier(
        hidden_layer_sizes = (42,),     # one hidden layer, 42 neurons
        activation         = 'tanh',    # hyperbolic tangent for hidden layer
        solver             = 'lbfgs',   # closest to Levenberg-Marquardt
        learning_rate_init = 0.1,       # as stated in paper
        max_iter           = 1000,      # enough for convergence
        random_state       = 42,        # reproducibility
        early_stopping     = False,
        verbose            = False
    )
    return model