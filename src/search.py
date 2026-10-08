"""Answer questions using context retrieved from the pgvector store."""

from dotenv import load_dotenv
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import PromptTemplate
from langchain_core.runnables import Runnable
from langchain_openai import ChatOpenAI
from langchain_postgres import PGVectorStore

from ingest import (
    DATABASE_URL,
    OPENAI_EMBEDDING_MODEL,
    PG_VECTOR_COLLECTION_NAME,
    VECTOR_SIZE,
    require_env,
    set_database_engine,
    set_embeddings,
)

load_dotenv()

OPENAI_CHAT_MODEL = require_env("OPENAI_CHAT_MODEL")
MODEL_TEMPERATURE = float(require_env("MODEL_TEMPERATURE"))

PROMPT_TEMPLATE = """
CONTEXTO:
{contexto}

REGRAS:
- Responda somente com base no CONTEXTO.
- Se a informação não estiver explicitamente no CONTEXTO, responda:
  "Não tenho informações necessárias para responder sua pergunta."
- Nunca invente ou use conhecimento externo.
- Nunca produza opiniões ou interpretações além do que está escrito.

EXEMPLOS DE PERGUNTAS FORA DO CONTEXTO:
Pergunta: "Qual é a capital da França?"
Resposta: "Não tenho informações necessárias para responder sua pergunta."

Pergunta: "Quantos clientes temos em 2024?"
Resposta: "Não tenho informações necessárias para responder sua pergunta."

Pergunta: "Você acha isso bom ou ruim?"
Resposta: "Não tenho informações necessárias para responder sua pergunta."

PERGUNTA DO USUÁRIO:
{pergunta}

RESPONDA A "PERGUNTA DO USUÁRIO"
"""


def build_vector_store() -> PGVectorStore:
    """Connect to the existing pgvector table that ingest.py populated."""
    engine = set_database_engine(url=DATABASE_URL)
    embeddings = set_embeddings(model=OPENAI_EMBEDDING_MODEL, vector_size=VECTOR_SIZE)
    return PGVectorStore.create_sync(
        engine=engine,
        table_name=PG_VECTOR_COLLECTION_NAME,
        embedding_service=embeddings,
    )


def build_llm() -> ChatOpenAI:
    """Build the chat model used to answer questions."""
    return ChatOpenAI(model=OPENAI_CHAT_MODEL, temperature=MODEL_TEMPERATURE)


def build_prompt() -> PromptTemplate:
    """Build the prompt template that wraps the retrieved context and the question."""
    return PromptTemplate(
        input_variables=["contexto", "pergunta"],
        template=PROMPT_TEMPLATE,
    )


def build_chain(prompt: PromptTemplate, llm: ChatOpenAI) -> Runnable:
    """Compose prompt, model and output parser into a single runnable chain."""
    if not isinstance(prompt, PromptTemplate):
        raise TypeError(f"Object of type {prompt.__class__} is not of type PromptTemplate")
    if not isinstance(llm, Runnable):
        raise TypeError(f"Object of type {llm.__class__} is not of type Runnable")
    return prompt | llm | StrOutputParser()


def retrieve_context(question: str, n_results: int, vector_store: PGVectorStore) -> str:
    """Return the text of the n_results chunks most similar to the question."""
    if not isinstance(n_results, int) or n_results <= 0:
        raise TypeError("n_results should be an int and greater than 0")     
    similar_documents = vector_store.similarity_search(question, k=int(n_results))
    return "\n".join(document.page_content.strip() for document in similar_documents)


def search_prompt(
    vector_store: PGVectorStore,
    question: str,
    n_results: int,
    chain: Runnable,
) -> str:
    """Retrieve context for the question and return the model's answer."""
    if not isinstance(question, str) or len(question.strip()) == 0:
        raise ValueError("Question is not valid. Should be of type str and length > 0")

    if not isinstance(vector_store, PGVectorStore):
        raise TypeError(f"Object of type {vector_store.__class__} is not of type PGVectorStore")

    if not isinstance(chain, Runnable):
        raise TypeError(f"Object of type {chain.__class__} is not of type Runnable")

    context = retrieve_context(
        question=question,
        n_results=n_results,
        vector_store=vector_store,
    )
    return chain.invoke({"contexto": context, "pergunta": question})