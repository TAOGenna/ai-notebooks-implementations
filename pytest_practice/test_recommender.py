import numpy as np
import recommender

def test_pick_top_k_uses_predict_proba(mocker):
    X = np.array([[0],[1],[2],[3]])
    fake_proba = np.array([
        [0.9, 0.1],
        [0.2, 0.8],
        [0.7, 0.3],
        [0.4, 0.6],
    ])

    model = mocker.Mock()
    model.predict_proba.return_value = fake_proba

    top = recommender.pick_top_k(model, X, k=2)

    # TODO #1: assert the model was called correctly
    assert model.assert_called_once_with()
    # TODO #2: assert the right indices were chosen (by class 1, desc)
    
