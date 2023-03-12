# ---
# jupyter:
#   jupytext:
#     formats: ipynb,py:percent
#     text_representation:
#       extension: .py
#       format_name: percent
#       format_version: '1.3'
#       jupytext_version: 1.14.5
#   kernelspec:
#     display_name: Python 3 (ipykernel)
#     language: python
#     name: python3
# ---

# %%
import csv
from metaflow import Flow, Step

flow_name = "GenerateCoOccurrenceVectorsFlow"
flow = Flow(flow_name)

# %%
id_to_name = {}

with open("yu-gi-oh-cards/data/cards.csv") as r:
    reader = csv.reader(r)
    next(reader)
    for idx, row in enumerate(reader):
        id_to_name[int(row[0])] = row[1]

# %%
from annoy import AnnoyIndex

model_ann = AnnoyIndex(20)
# from tempfile import NamedTemporaryFile
# with NamedTemporaryFile() as tmp: #D
#     tmp.write(Step(f"{flow_name}/{flow.latest_run.id}/end").task.data.model_ann) #D
model_ann.load("/var/folders/26/kknv0qdn23x8cn86y7p06j4r0000gp/T/tmptkswylhx")  # D

# %%
selected = 9822220

# %%
dark_magician_vec = model_ann.get_item_vector(46986414)

# %%
cards, distances = model_ann.get_nns_by_item(selected, 10, search_k=-1, include_distances=True)

# %%
for card_id, distance in zip(cards, distances):
    print(id_to_name[card_id])

# %%
