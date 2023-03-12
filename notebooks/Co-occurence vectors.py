import csv
from pathlib import Path
import numpy as np
from scipy.sparse import coo_matrix

# +
# # !git clone --single-branch --depth 1 https://github.com/fferegrino/yu-gi-oh.git yu-gi-oh-cards

# +
card_to_id = {}
id_to_card = {}

with open('yu-gi-oh-cards/data/cards.csv') as r:
    reader = csv.reader(r)
    next(reader)
    for idx, row in enumerate(reader):
        card_to_id[int(row[0])] = idx
        id_to_card[idx] = int(row[0])


with open('yu-gi-oh-cards/data/cards_variants.csv') as r:
    reader = csv.reader(r)
    next(reader)
    for idx, row in enumerate(reader):
        original = int(row[0])
        variant = int(row[1])
        if variant not in card_to_id:
            try:
                card_to_id[variant] = card_to_id[original]
            except:
                print(f'Original {original}')
        else:
            print(f'{variant} already exists in the dict')
# -

len(card_to_id), len(id_to_card)

# # Download latest decks from GitHub

# +
# # !git clone --single-branch --depth 1 https://github.com/fferegrino/yu-gi-oh-decks.git
# -

NUMBER_CARDS = len(id_to_card)

# Create a co-occurence matrix with a Numpy array
coo = np.zeros((NUMBER_CARDS, NUMBER_CARDS), dtype=np.int32)


# +


csv_file = './yu-gi-oh-decks/data/0000000.csv'

MAIN_DECK_ROW = 6
EXTRA_DECK_ROW = 7
SIDE_DECK_ROW = 8

with open(csv_file) as r:
    reader = csv.reader(r)
    next(reader)
    for row in reader:
        cards_in_deck = []
        for col in [MAIN_DECK_ROW, EXTRA_DECK_ROW, SIDE_DECK_ROW]:
            cards_ = [
                int(card_str.strip("\"")) for card_str in row[col][1:-1].split(',') if card_str
            ]
            cards_in_deck.extend(cards_)

        for card in cards_in_deck:
            for other_card in cards_in_deck:
                if card == other_card:
                    try:
                        coo[card_to_id[card], card_to_id[other_card]] += 1
                    except:
                        print(f"{card} {other_card}")
            
# -
coo_matrix(coo)



