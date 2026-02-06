from pathlib import Path
import logging

from ..types import EmbeddingClient, QueryModel
from .etl import bronze_database, silver_database, gold_database


logger = logging.getLogger(__name__)
def build_database(
        documents_path: Path,
        database_path: Path,
        embedder: EmbeddingClient,
        context_model: QueryModel,
        file_path_annotations_config: dict,
        force_rebuild: bool
    ):
    
    bronze_path = database_path / "medallions" / "bronze.parquet"
    silver_path = database_path / "medallions" / "silver.parquet"
    gold_path = database_path / "medallions" / "gold.parquet"

    database_path.mkdir(parents=True, exist_ok=True)
    bronze_path.parent.mkdir(parents=True, exist_ok=True)


    if not documents_path.exists():
        raise FileNotFoundError(f"Documents path {documents_path} does not exist")

    df_exists = lambda p: p.exists() and p.stat().st_size > 0
   
    # Extract the text from the input files and store it in a parquet file
    # Expect columns: file_path, content
    if force_rebuild or (not df_exists(bronze_path) and not df_exists(silver_path) and not df_exists(gold_path)):
        logger.info("Building bronze database...")
        bronze_database(documents_path, bronze_path)
        logger.info("Bronze database built successfully.")
    else:
        logger.info("Bronze database already exists or is not required. Skipping construction...")

    # Chunk the text and store it in a parquet file. Add context to the chunks.
    # Expect columns: idx, file_path, chunk_index, file_path_annotations, contextual_annotations, chunk_text
    if force_rebuild or (not df_exists(silver_path) and not df_exists(gold_path)):
        logger.info("Building silver database...")
        silver_database(bronze_path, silver_path, file_path_annotations_config, context_model, database_path)
        logger.info("Silver database built successfully.")
    else:
        logger.info("Silver database already exists or is not required. Skipping construction...")

    # Take fully processed data, add vectors, and store it in a parquet file.
    # Expect columns: idx, file_path, chunk_index, file_path_annotations, contextual_annotations, chunk_text, embedding
    if force_rebuild or not df_exists(gold_path):
        logger.info("Building gold database...")
        gold_database(silver_path, gold_path, embedder, database_path)
        logger.info("Gold database built successfully.")
    else:
        logger.info("Gold database already exists. Skipping construction...")