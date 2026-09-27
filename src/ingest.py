"""Ingest a PDF into a Postgres pgvector store for semantic search."""

import hashlib
import os
from pathlib import Path

from dotenv import load_dotenv
from langchain_community.document_loaders import PyPDFLoader
from langchain_core.documents import Document
from langchain_openai import OpenAIEmbeddings
from langchain_postgres import Column, PGEngine, PGVectorStore
from langchain_text_splitters import RecursiveCharacterTextSplitter
from sqlalchemy.exc import ProgrammingError

load_dotenv()


def require_env(name: str) -> str:
    """Return the named environment variable, or exit with an actionable message."""
    value = os.getenv(name)
    if not value:
        raise SystemExit(f"Missing required environment variable: {name}")
    return value


DATABASE_URL = require_env("DATABASE_URL")
PG_VECTOR_COLLECTION_NAME = require_env("PG_VECTOR_COLLECTION_NAME")
OPENAI_EMBEDDING_MODEL = require_env("OPENAI_EMBEDDING_MODEL")
VECTOR_SIZE = int(require_env("VECTOR_SIZE"))
if VECTOR_SIZE <= 0:
    raise SystemExit("VECTOR_SIZE must be a positive integer")

PDF_PATH = Path.cwd() / os.getenv("PDF_PATH", "")


def load_pdf_file(file_path: str | os.PathLike) -> list[Document]:
    """Load a PDF file and return its extracted pages as documents."""
    if not Path(file_path).is_file():
        raise FileNotFoundError(file_path)
    loader = PyPDFLoader(file_path)
    return loader.load()


def text_splitter(documents: list[Document]) -> list[Document]:
    """Split documents into overlapping chunks, dropping empty metadata values."""
    processor = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=150)
    return [
        Document(
            page_content=chunk.page_content,
            metadata={k: v for k, v in chunk.metadata.items() if v not in (None, "")},
        )
        for chunk in processor.split_documents(documents)
    ]


def set_database_engine(url: str) -> PGEngine:
    """Build the PGEngine used for both table setup and the vector store."""
    return PGEngine.from_connection_string(url=url)


def ensure_vectorstore_table(engine: PGEngine, table_name: str, vector_size: int) -> None:
    """Create the pgvector table if it doesn't already exist.

    When a table was created on a previous run PGVector throws a duplicate relation error.
    Checking that the table exists first is mandatory. PGEngine doesn't an inspect method,
    therefore an exception was implemented to handle that
    """
    try:
        engine.init_vectorstore_table(
            table_name=table_name,
            vector_size=vector_size,
            id_column=Column(name="langchain_id", data_type="VARCHAR"),
        )
    except ProgrammingError:
        pass


def set_embeddings(model: str, vector_size: int) -> OpenAIEmbeddings:
    """Build the embedding model used to vectorize chunks."""
    return OpenAIEmbeddings(model=model, dimensions=vector_size)


def build_chunk_id(document: Document) -> str:
    """Derive a stable id from a chunk's content.

    Hashing content (rather than position) to allow upserts while keeping uniqueness of keys.
    """
    digest = hashlib.sha256(document.page_content.encode("utf-8")).hexdigest()
    return f"doc-{digest}"


def ingest_pdf(
    file_path: str | os.PathLike,
    vector_size: int,
    table_name: str,
    engine: PGEngine,
    embeddings: OpenAIEmbeddings,
) -> None:
    """Load, chunk, and upsert a PDF's contents into the vector store."""
    documents = load_pdf_file(file_path)
    chunked_document = text_splitter(documents)
    ids = [build_chunk_id(document) for document in chunked_document]

    ensure_vectorstore_table(engine=engine, table_name=table_name, vector_size=vector_size)

    vector_store = PGVectorStore.create_sync(
        engine=engine,
        table_name=table_name,
        embedding_service=embeddings,
    )
    vector_store.add_documents(documents=chunked_document, ids=ids)


if __name__ == "__main__":
    engine = set_database_engine(url=DATABASE_URL)
    embeddings = set_embeddings(model=OPENAI_EMBEDDING_MODEL, vector_size=VECTOR_SIZE)
    ingest_pdf(
        PDF_PATH,
        engine=engine,
        embeddings=embeddings,
        vector_size=VECTOR_SIZE,
        table_name=PG_VECTOR_COLLECTION_NAME,
    )
