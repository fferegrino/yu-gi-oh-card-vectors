MAIN_DECK_ROW = 6
EXTRA_DECK_ROW = 7
SIDE_DECK_ROW = 8


def fill_matrix_from_file(csv_file, card_to_id, coo, cards_not_in_index):
    import csv

    with open(csv_file) as r:
        reader = csv.reader(r)
        next(reader)
        for row in reader:
            cards_in_deck = []
            for col in [MAIN_DECK_ROW, EXTRA_DECK_ROW, SIDE_DECK_ROW]:
                cards_ = [int(card_str.strip('"')) for card_str in row[col][1:-1].split(",") if card_str.strip('"')]
                cards_in_deck.extend(cards_)

            for card in cards_in_deck:
                try:
                    origin_card = card_to_id[card]
                except KeyError:
                    if card not in cards_not_in_index:
                        cards_not_in_index.add(card)
                    continue

                for other_card in cards_in_deck:
                    try:
                        destination_card = card_to_id[other_card]
                    except KeyError:
                        if other_card not in cards_not_in_index:
                            cards_not_in_index.add(other_card)
                        continue

                    coo[origin_card, destination_card] += 1
