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
from metaflow import Flow, Step

flow = Flow("CoOcurrenceMatrixFlow")

# %%
flow.latest_run.id


# %%
list(flow.latest_run.steps())

# %%
co_occurrence_matrix = Step(f"CoOcurrenceMatrixFlow/{flow.latest_run.id}/end").task.data.matrix.astype(float)


# %%
co_occurrence_matrix

# %%
from scipy.sparse.linalg import svds
import numpy as np

# %%
U, S, Vt = svds(co_occurrence_matrix, k=3)

# %%
embeddings = U * np.sqrt(S)

# Normalize embeddings
embeddings = (embeddings - np.mean(embeddings, axis=0)) / np.std(embeddings, axis=0)

# %%
embeddings

# %%
import tensorflow as tf
from tensorboard.plugins import projector
import numpy as np

# Set up a TensorFlow summary writer
log_dir = "logs/embeddings"
summary_writer = tf.summary.create_file_writer(log_dir)

# Create the embedding variable
embedding_var = tf.Variable(embeddings, name="embedding")

# Configure the projector
config = projector.ProjectorConfig()
embedding = config.embeddings.add()
embedding.tensor_name = embedding_var.name

# Write the metadata to a file
metadata_path = f"{log_dir}/metadata.tsv"
with open(metadata_path, "w") as metadata_file:
    metadata_file.write("\n".join(["label_{}".format(i) for i in range(len(embeddings))]))

embedding.metadata_path = metadata_path

# Write the projector to disk
with summary_writer.as_default():
    tf.summary.scalar("my_scalar", 0.5, step=0)
    tf.summary.tensor_embedding(config)

# Close the summary writer
summary_writer.close()
summary_writer.close()

# %%
