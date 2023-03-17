from itertools import permutations
from collections import Counter
import csv

MAIN_DECK_ROW = 6
EXTRA_DECK_ROW = 7
SIDE_DECK_ROW = 8


def fill_matrix_from_file(csv_file, card_to_id, coo):
    with open(csv_file) as r:
        reader = csv.reader(r)
        next(reader)
        for row in reader:
            cards_in_deck = []
            for col in [MAIN_DECK_ROW, EXTRA_DECK_ROW, SIDE_DECK_ROW]:
                cards = row[col][1:-1].split(",")
                cards = [int(card_str.strip('"')) for card_str in cards if card_str.strip('"')]
                cards = [card for card in cards if card in card_to_id]

                cards_in_deck.extend(cards)

            fill_matrix_with_all_combinations(coo, cards_in_deck, card_to_id)


def fill_matrix_with_all_combinations(coo_matrix, deck_passcodes, passcode_to_id):
    all_combinations = Counter(permutations(deck_passcodes, 2))
    for (card1, card2), count in all_combinations.items():
        coo_matrix[passcode_to_id[card1], passcode_to_id[card2]] += count
