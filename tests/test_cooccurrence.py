from cooccurrence import fill_matrix_with_all_combinations
import numpy as np


def test_fill_matrix_with_all_combinations():
    coo_matrix = np.zeros((3, 3), dtype=np.int32)
    deck_passcodes = [1, 2, 2, 3]
    passcode_to_id = {1: 0, 2: 1, 3: 2}
    fill_matrix_with_all_combinations(coo_matrix, deck_passcodes, passcode_to_id)

    expected = [(1, 2), (1, 2), (1, 3), (2, 1), (2, 2), (2, 3), (2, 1), (2, 2), (2, 3), (3, 1), (3, 2), (3, 2)]

    arr = np.zeros((3, 3), dtype=np.int32)

    for card1, card2 in expected:
        arr[passcode_to_id[card1], passcode_to_id[card2]] += 1

    np.testing.assert_array_equal(coo_matrix, arr)
