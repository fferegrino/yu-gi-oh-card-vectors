MAIN_DECK_ROW = 6
EXTRA_DECK_ROW = 7
SIDE_DECK_ROW = 8


def fill_matrix_from_file(csv_file, card_to_id, coo):
    import csv
    from itertools import combinations
    from collections import Counter

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

            all_combinations = Counter(combinations(cards_in_deck, 2))
            for (card1, card2), count in all_combinations.items():
                coo[card_to_id[card1], card_to_id[card2]] += count
