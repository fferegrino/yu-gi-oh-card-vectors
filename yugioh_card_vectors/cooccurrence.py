from itertools import permutations
from collections import Counter
import csv

MAIN_DECK_COLUMN = 6
EXTRA_DECK_COLUMN = 7
SIDE_DECK_COLUMN = 8


def fill_matrix_from_file(csv_file, passcode_to_id, coo):
    with open(csv_file) as r:
        reader = csv.reader(r)
        next(reader)
        for row in reader:
            cards_in_deck = []
            for col in [MAIN_DECK_COLUMN, EXTRA_DECK_COLUMN, SIDE_DECK_COLUMN]:
                card_ids = row[col][1:-1].split(",")
                card_ids = [int(card_str.strip('"')) for card_str in card_ids if card_str.strip('"')]
                card_ids = [passcode_to_id[passcode] for passcode in card_ids if passcode in passcode_to_id]

                cards_in_deck.extend(card_ids)

            fill_matrix_with_all_combinations(coo, cards_in_deck)


def fill_matrix_with_all_combinations(coo_matrix, cards_in_deck):
    all_combinations = Counter(permutations(cards_in_deck, 2))
    for (card1_id, card2_id), count in all_combinations.items():
        coo_matrix[card1_id, card2_id] += count
