

from metaflow import FlowSpec, step


MAIN_DECK_ROW = 6
EXTRA_DECK_ROW = 7
SIDE_DECK_ROW = 8

def fill_matrix_from_file(csv_file, card_to_id, coo):
    import csv
    with open(csv_file) as r:
        reader = csv.reader(r)
        next(reader)
        for row in reader:
            cards_in_deck = []
            for col in [MAIN_DECK_ROW, EXTRA_DECK_ROW, SIDE_DECK_ROW]:
                cards_ = [
                    int(card_str.strip("\"")) for card_str in row[col][1:-1].split(',') if card_str.strip("\"")
                ]
                cards_in_deck.extend(cards_)

            for card in cards_in_deck:
                for other_card in cards_in_deck:
                    if card == other_card:
                        try:
                            coo[card_to_id[card], card_to_id[other_card]] += 1
                        except:
                            print(f"{card} {other_card}")



class CoOcurrenceMatrixFlow(FlowSpec):
    @step
    def start(self):
        self.cards_repo = 'https://github.com/fferegrino/yu-gi-oh.git'
        self.decks_repo = 'https://github.com/fferegrino/yu-gi-oh-decks.git'
        self.next(self.build_reference_dicts)

    @step
    def build_reference_dicts(self):
        import subprocess
        import tempfile
        import csv

        with tempfile.TemporaryDirectory() as tmpdirname:
            subprocess.run(['git', 'clone', '--single-branch', '--depth', '1',
                            self.cards_repo, tmpdirname])

            self.card_to_id = {}
            self.id_to_card = {}

            with open(tmpdirname + '/data/cards.csv') as r:
                reader = csv.reader(r)
                next(reader)
                for idx, row in enumerate(reader):
                    self.card_to_id[int(row[0])] = idx
                    self.id_to_card[idx] = int(row[0])

            with open(tmpdirname + '/data/cards_variants.csv') as r:
                reader = csv.reader(r)
                next(reader)
                for idx, row in enumerate(reader):
                    original = int(row[0])
                    variant = int(row[1])
                    if variant not in self.card_to_id:
                        try:
                            self.card_to_id[variant] = self.card_to_id[original]
                        except:
                            print(f'Error with id {original}')
        self.card_count = len(self.id_to_card)
        self.next(self.build_cooccurence_matrix)

    @step
    def build_cooccurence_matrix(self):
        import subprocess
        import tempfile
        from scipy.sparse import coo_matrix
        import numpy as np
        import glob

        coo = np.zeros((self.card_count, self.card_count), dtype=np.int32)

        with tempfile.TemporaryDirectory() as tmpdirname:
            subprocess.run(['git', 'clone', '--single-branch', '--depth', '1',
                            self.decks_repo, tmpdirname])

            for csv_file in glob.glob(tmpdirname + '/data/*.csv'):
                fill_matrix_from_file(csv_file, self.card_to_id, coo)

        self.cooccurence_matrix = coo
        self.matrix = coo_matrix(coo)

        self.next(self.end)

    @step
    def end(self):
        pass

if __name__ == '__main__':
    CoOcurrenceMatrixFlow()