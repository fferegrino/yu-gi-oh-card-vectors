from metaflow import FlowSpec, step, Parameter, profile, conda, resources, S3


class GenerateCoOccurrenceVectorsFlow(FlowSpec):
    """ """

    embedding_size = Parameter("embedding_size", default=20, help="Size of the embedding vectors")
    annoy_trees = Parameter("annoy_trees", default=5, help="Number of trees to use in the Annoy index")
    preserve_annoy_index = Parameter("preserve_annoy_index", default=False, help="Preserve the Annoy index")

    @step
    def start(self):
        """
        Setups the flow with the appropriate repository URLs
        """
        self.cards_repo = "https://github.com/fferegrino/yu-gi-oh.git"
        self.decks_repo = "https://github.com/fferegrino/yu-gi-oh-decks.git"
        self.next(self.build_reference_dicts)

    @step
    def build_reference_dicts(self):
        """
        Builds the reference dictionaries for the cards, these will map the card ID to the index in the embedding matrix
        """
        from git import clone_repo
        import csv

        with clone_repo(self.cards_repo) as (repo_path, commit_hash):
            self.cards_commit_hash = commit_hash
            self.card_to_id = {}
            self.id_to_card = {}

            with open(repo_path + "/data/cards.csv") as r:
                reader = csv.reader(r)
                next(reader)
                for idx, row in enumerate(reader):
                    self.card_to_id[int(row[0])] = idx
                    self.id_to_card[idx] = int(row[0])

            with open(repo_path + "/data/cards_variants.csv") as r:
                reader = csv.reader(r)
                next(reader)
                for idx, row in enumerate(reader):
                    original = int(row[0])
                    variant = int(row[1])
                    if variant not in self.card_to_id:
                        try:
                            self.card_to_id[variant] = self.card_to_id[original]
                        except:
                            print(f"Error with id {original}")
        self.card_count = len(self.id_to_card)
        self.next(self.build_cooccurrence_matrix)

    @conda(libraries={"scipy": "1.10.1", "numpy": "1.24.2"})
    @step
    def build_cooccurrence_matrix(self):
        """
        Build a sparse co-occurrence matrix for the cards in the decks
        """
        from git import clone_repo
        from scipy.sparse import coo_matrix
        import numpy as np
        import glob
        from cooccurrence import fill_matrix_from_file

        coo = np.zeros((self.card_count, self.card_count), dtype=np.int32)

        self.cards_not_in_index = set()

        with clone_repo(self.decks_repo) as (repo_path, commit_hash):
            self.decks_commit_hash = commit_hash
            for csv_file in glob.glob(repo_path + "/data/*.csv"):
                fill_matrix_from_file(csv_file, self.card_to_id, coo, self.cards_not_in_index)

        self.matrix = coo_matrix(coo)

        self.next(self.build_embeddings)

    @conda(libraries={"scipy": "1.10.1", "numpy": "1.24.2"})
    @step
    def build_embeddings(self):
        """
        Build the embeddings using the co-occurrence matrix with the SVD algorithm
        """
        from scipy.sparse.linalg import svds
        import numpy as np

        u, s, _ = svds(self.matrix.astype(float), k=self.embedding_size)
        embeddings = u * np.sqrt(s)

        # Normalize embeddings
        self.embeddings = (embeddings - np.mean(embeddings, axis=0)) / np.std(embeddings, axis=0)

        self.next(self.build_index)

    @conda(libraries={"python-annoy": "1.17.1", "numpy": "1.24.2"})
    @resources(memory=16000)
    @step
    def build_index(self):
        """
        Build the Annoy index for the embeddings
        """
        from annoy import AnnoyIndex
        from tempfile import NamedTemporaryFile

        with NamedTemporaryFile(delete=False) as tmp:
            ann = AnnoyIndex(self.embedding_size, "angular")
            ann.on_disk_build(tmp.name)
            with profile("Add vectors"):
                for idx, card_vector in enumerate(self.embeddings):
                    ann.add_item(self.id_to_card[idx], card_vector)
            with profile("Build index"):
                ann.build(self.annoy_trees)

            print(f"Writing index to {tmp.name} and not deleting")

            with S3(run=self) as s3:
                upload_result = s3.put_files([("index.ann", tmp.name)])
                self.index_s3_url = upload_result[0][1]
                print("Object saved at", self.index_s3_url)

        self.next(self.end)

    @step
    def end(self):
        import boto3

        s3 = boto3.resource("s3")
        bucket, key = self.index_s3_url.replace("s3://", "").split("/", 1)
        copy_source = {"Bucket": bucket, "Key": key}
        bucket = s3.Bucket("feregrino-metaflow-experiments")
        bucket.copy(copy_source, "yu-gi-oh/index.ann")


if __name__ == "__main__":
    GenerateCoOccurrenceVectorsFlow()
