from pathlib import Path
import logging
import re
from typing import Dict, List
from concurrent.futures import ThreadPoolExecutor, as_completed
import os

import polars as pl
from docling.document_converter import DocumentConverter
from tqdm import tqdm
import chonkie
from chonkie import SentenceChunker
import numpy as np

from ..types import EmbeddingClient, QueryModel
from ..query.query_manager import QueryManager
from ..query.embedding_manager import EmbeddingManager

logger = logging.getLogger(__name__)

def bronze_database(documents_path: Path, bronze_path: Path):
    """
    Extract text from deeply nested documents using docling.
    
    Walks recursively through documents_path and uses docling to extract text
    from all supported document types. Fails hard if any document cannot be parsed.
    
    Args:
        documents_path: Path to directory containing documents (supports nested directories)
        bronze_path: Path where bronze parquet file will be saved
        
    Returns:
        None (saves parquet file to bronze_path)
        
    Raises:
        Exception: If any document cannot be parsed by docling
    """    
    # Ensure output directory exists
    bronze_path.parent.mkdir(parents=True, exist_ok=True)
    
    converter = DocumentConverter()
    results = []
    
    # Collect all files first so tqdm can show total progress
    all_files = [p for p in documents_path.rglob("*") if p.is_file()]
    
    # Walk recursively through all files in documents_path with progress tracking
    for file_path in tqdm(all_files, desc="Building bronze database", unit="file"):
        try:
            logger.debug(f"Processing {file_path}...")
            # Convert document using docling
            result = converter.convert(str(file_path))
            # Extract text as markdown
            content = result.document.export_to_markdown()
            
            # Store relative path from documents_path for consistency
            relative_path = file_path.relative_to(documents_path)
            results.append({
                "file_path": str(relative_path),
                "content": content
            })
            logger.debug(f"Successfully processed {file_path}")
        except Exception as e:
            # Fail hard on any unparsable document
            error_msg = f"Failed to parse document {file_path}: {str(e)}"
            logger.error(error_msg)
            raise RuntimeError(error_msg) from e
    
    if not results:
        logger.warning(f"No documents found in {documents_path}")
        # Create empty dataframe with correct schema
        df = pl.DataFrame({
            "file_path": [],
            "content": []
        }, schema={"file_path": pl.Utf8, "content": pl.Utf8})
    else:
        # Create dataframe and save to parquet
        df = pl.DataFrame(results)
    
    df.write_parquet(bronze_path)
    logger.info(f"Bronze database saved to {bronze_path} with {len(df)} documents")


def silver_database(
        bronze_path: Path,
        silver_path: Path,
        file_path_annotations_config: dict,
        context_model: QueryModel,
        database_path: Path
    ):
    bronze = pl.read_parquet(bronze_path)

    records = bronze.to_dicts()

    qm = QueryManager(context_model)

    def process_record(record: dict) -> list[dict]:
        # Initialize chunker. Target approximately 500 tokens, no overlap
        chunker = SentenceChunker(
            chunk_size=2048,
            chunk_overlap=0,
            min_sentences_per_chunk=1,
            min_characters_per_sentence=10,
        )

        text = record.get("content", "") or ""

        # Strip continuous strings of non-alphanumeric characters
        pat = re.compile(r"[\W^_]{5,}", flags=re.UNICODE)
        text = pat.sub(" ", text)

        # Fast path for empty text
        if not text.strip():
            return []

        chunks = _normalize_chunks(list(chunker(text)))
        results: list[dict] = []

        for idx, ch in enumerate(chunks):
            out = {
                "file_path": record["file_path"],
                "chunk_index": idx,
                "chunk_text": ch,
                "file_path_annotations": _file_path_annotations(
                    record["file_path"],
                    file_path_annotations_config,
                ),
            }
            try:
                out["contextual_annotations"] = _contextual_annotations(
                    ch,
                    idx,
                    record,
                    qm,
                    database_path
                )
            except Exception as exp:
                logger.error(f"Error annotating chunk: {exp}")
                out["contextual_annotations"] = ""
            results.append(out)
        return results
    # Multithread processing of rows
    max_workers = 50
    output_rows: list[dict] = []
    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        futures = [executor.submit(process_record, rec) for rec in records]
        for f in tqdm(
            as_completed(futures), total=len(futures), desc="Chunking rows", unit="row"
        ):
            try:
                output_rows.extend(f.result())
            except Exception as e:
                # Skip problematic rows but continue processing
                logger.error(f"Error processing row: {e}")
                continue

    if not output_rows:
        raise Exception("Failed to generate any output columns.")

    return pl.DataFrame(output_rows)
    

def _normalize_chunks(chunks: List[chonkie.Chunk]) -> list[str]:
    """
    Normalize chunks by removing empty or junk chunks and converting to strings.

    - chunks: list of chunks

    Returns a list of strings
    """
    normalized_chunks = []
    for chunk in chunks:
        chunk = chunk.text.strip()
        if not chunk:
            continue
        if len(chunk) < 100 and normalized_chunks:
            normalized_chunks[-1] += chunk
            continue
        normalized_chunks.append(chunk)
    return normalized_chunks

def _file_path_annotations(
    file_path: str,
    file_path_annotation_config: Dict[str, str],
) -> str:
    """
    Annotate chunks based on their file path using regex patterns.

    - file_path: file path of the original file
    - file_path_annotation_config: mapping of regex pattern -> annotation text (string only)

    Returns the annotated chunk text
    """
    annotations = []
    for pattern, annotation_text in file_path_annotation_config.items():
        if re.search(pattern, file_path):
            annotations.append(annotation_text)
    return "\n".join(annotations)

def _contextual_annotations(
    ch: str, idx: int, record: dict, qm: QueryManager, database_path: Path
) -> str:
    """
    Get contextual information about a chunk and its place in a file.

    - ch: chunk text
    - idx: chunk index
    - record: bronze medallion record
    - context_model: query model

    Returns the contextualized chunk text
    """
    file_path = record["file_path"]
    whole_document = record["content"]

    # Check the cache
    cache_path = (
        database_path
        / "context"
        / "query_cache"
        / qm.query_model.model_id
        / Path(file_path).with_suffix("")
        / f"{idx}.txt"
    )
    cache_path.parent.mkdir(parents=True, exist_ok=True)
    if cache_path.exists() and not cache_path.stat().st_size == 0:
        with open(cache_path, "r") as f:
            return f.read()

    if whole_document is None:
        raise ValueError(f"Whole document not found for file path: {file_path}")
    query = f"""<document>
{whole_document}
</document>
Here is the chunk we want to situate within the whole document
<chunk>
{ch}
</chunk>
Please give a short succinct context to situate this chunk within the overall document for the purposes of improving search retrieval of the chunk. Answer only with the succinct context and nothing else.
"""
    result = qm.query(query)
    with open(cache_path, "w") as f:
        f.write(result)
    return result

def gold_database(silver_path: Path, gold_path: Path, embedder: EmbeddingClient, database_path: Path):
    gold = pl.read_parquet(silver_path)

    # Create EmbeddingManager to enforce rate limits
    embedding_manager = EmbeddingManager(embedder)

    # Combine annotations and chunk text for embedding
    gold = gold.with_columns(
        contextual_chunk=pl.concat_str(
            [
                pl.col("file_path_annotations"),
                pl.col("contextual_annotations"),
                pl.col("chunk_text"),
            ],
            separator="\n\n",
            ignore_nulls=True,
        )
    )
    
    def embed_chunk(record_with_idx: tuple[int, dict]) -> tuple[int, list[float]]:
        """Embed a single chunk, returning (index, embedding)."""
        idx, record = record_with_idx
        cache_path = database_path / "embeddings" / Path(record["file_path"]).with_suffix("") / f"{record['chunk_index']}.npy"
        if cache_path.exists() and not cache_path.stat().st_size == 0:
            embedding = np.load(cache_path).tolist()
        else:
            embedding = embedding_manager.embed(record["contextual_chunk"]).tolist()
        
        return (idx, embedding)
    
    # Multithreaded embedding generation
    records = gold.to_dicts()
    max_workers = 50
    embeddings_dict: dict[int, list[float]] = {}
    
    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        futures = [executor.submit(embed_chunk, (idx, rec)) for idx, rec in enumerate(records)]
        for f in tqdm(
            as_completed(futures), total=len(futures), desc="Generating embeddings", unit="chunk"
        ):
            try:
                idx, embedding = f.result()
                embeddings_dict[idx] = embedding
            except Exception as e:
                logger.error(f"Error embedding chunk: {e}")
                continue
    
    # Add embeddings back to dataframe
    if embeddings_dict:
        # Sort by index to maintain order
        sorted_indices = sorted(embeddings_dict.keys())
        embeddings_list = [embeddings_dict[idx] for idx in sorted_indices]
        gold = gold.with_columns(
            embedding=pl.Series(embeddings_list, dtype=pl.List(pl.Float32))
        )
    else:
        raise Exception("Failed to generate any embeddings.")
    
    gold.write_parquet(gold_path)
