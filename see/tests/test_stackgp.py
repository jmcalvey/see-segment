
import numpy as np
import StackGP as sgp


def test_stackgp():
    # Generate sample data from the official example
    np.random.seed(42)

    pts = 100

    inputs = np.array([
        np.random.normal(1, 5, pts),
        np.random.normal(1, 2, pts),
        np.random.normal(3, 10, pts)
    ])

    response = inputs[0] / (
        1.3 - inputs[1] / inputs[2]
    )

    # Evolve models
    models = sgp.evolve(
        inputs,
        response,
        generations=10,
        tracking=False,
        ops=sgp.allOps()
    )

    # Check that evolution returned models
    assert len(models) > 0

    best_model = models[0]

    # Evaluate the best model
    predictions = sgp.evaluateGPModel(
        best_model, inputs
    )

    # Check output dimensions
    assert np.asarray(predictions).shape == response.shape

    # Display results
    print("\nBest model:")
    print(sgp.printGPModel(best_model))
    print("Fitness:", best_model[2][0])
