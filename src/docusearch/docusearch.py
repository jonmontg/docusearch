from pathlib import Path
import gc
import yaml

import polars as pl

from .types import EmbeddingClient, QueryModel
from .database.build_database import build_database
from .search.search_client import SearchClient

class Docusearch:
    def __init__(self,
        documents_path: str | Path,
        database_path: str | Path,
        embedder: EmbeddingClient,
        context_model: QueryModel,
        file_path_annotations_config: str | Path = None,
        force_rebuild: bool = False,
        vector_normalize: bool = True,
    ):
        self.documents_path = Path(documents_path)
        self.database_path = Path(database_path)
        self.embedder = embedder

        if file_path_annotations_config is not None:
            with open(file_path_annotations_config, "r") as f:
                file_path_annotations_config = yaml.safe_load(f)
        else:
            file_path_annotations_config = {}

        build_database(
            self.documents_path,
            self.database_path,
            self.embedder,
            context_model,
            file_path_annotations_config,
            force_rebuild
        )

        # Build search client
        lazy_df = pl.scan_parquet(self.database_path / "medallions" / "gold.parquet")
        embeddings = (
            lazy_df.select(
                pl.col("embedding").cast(pl.Array(pl.Float32, embedder.dimensions)).alias("emb")
            )
            .collect()
            .to_series()
        ).to_numpy()
        chunk_df = lazy_df.drop("embedding").collect()
        gc.collect()

        self.search_client = SearchClient(chunk_df, embeddings, self.embedder, vector_normalize)

    def search(self, query: str, k: int = 10):
        return self.search_client.search(query, k)