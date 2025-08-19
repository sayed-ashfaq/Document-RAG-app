import sys
import os
from operator import itemgetter
from typing import Optional, List, Dict, Any

from langchain_core.output_parsers import StrOutputParser
from langchain_core.messages import BaseMessage
from langchain_core.prompts import ChatPromptTemplate
from langchain_community.vectorstores import FAISS

from utils.model_loader import ModelLoader
from logger.custom_logger import CustomLogger
from exception.custom_exception_archive import DocumentPortalException
from prompts.prompt_library import PROMPT_REGISTRY
from model.models import PromptType



class ConversationalRAG:
    """
    Conversational Retrieval-Augmented Generation (RAG) system.

    This class provides an interface for building and invoking a
    conversational RAG pipeline using FAISS for vector storage,
    LangChain for chaining, and a Large Language Model (LLM).

    Responsibilities:
        - Initialize and manage the LLM.
        - Load a FAISS retriever for document search.
        - Build a LangChain Expression Language (LCEL) graph for
          contextual question answering.
        - Provide a public `invoke` method for user queries with
          chat history context.

    Attributes:
        session_id (Optional[str]): Unique identifier for the session (used for logging).
        retriever: FAISS retriever object for document retrieval.
        chain: LCEL pipeline that ties together contextualization, retrieval, and answer generation.
        llm: Loaded Large Language Model instance.
        contextualize_prompt (ChatPromptTemplate): Prompt for rewriting user questions.
        qa_prompt (ChatPromptTemplate): Prompt for final question answering.
        log: Custom logger instance.
    """

    def __init__(self, session_id: Optional[str], retriever=None):
        """
        Initialize a ConversationalRAG instance.

        Args:
            session_id (Optional[str]): Unique session identifier.
            retriever: Optional retriever instance. If provided, LCEL chain
                will be built immediately.

        Raises:
            DocumentPortalException: If initialization fails.
        """
        try:
            self.log = CustomLogger().get_logger(__name__)
            self.session_id = session_id

            # Load LLM and prompts at once
            self.llm = self._load_llm()
            self.contextualize_prompt: ChatPromptTemplate = PROMPT_REGISTRY[
                PromptType.CONTEXTUALIZE_QUESTION.value
            ]
            self.qa_prompt: ChatPromptTemplate = PROMPT_REGISTRY[
                PromptType.CONTEXT_QA.value
            ]

            # Lazy pieces
            self.retriever = retriever
            self.chain = None

            if self.retriever is not None:
                self._build_lcel_chain()

            self.log.info(
                "ConversationalRAG initialized successfully",
                session_id=self.session_id
            )
        except Exception as e:
            self.log.error(f"Error initializing ConversationalRAG: {e}")
            raise DocumentPortalException("Error initializing ConversationalRAG", sys)

    # ----------------Public API----------------------#
    def load_retriever_from_faiss(
            self,
            index_path: str,
            k: int = 5,
            index_name: str = "index",
            search_type: str = "similarity",
            search_kwargs: Optional[Dict[str, Any]] = None,
    ):
        """
        Load FAISS retriever from local index files.

        Args:
            index_path (str): Path to the FAISS index directory.
            k (int, optional): Number of nearest neighbors to retrieve. Defaults to 5.
            index_name (str, optional): Name of the FAISS index. Defaults to "index".
            search_type (str, optional): Retrieval strategy ("similarity", "mmr", etc.).
            search_kwargs (Optional[Dict[str, Any]]): Extra parameters for the retriever.

        Returns:
            retriever: Loaded FAISS retriever instance.

        Raises:
            FileNotFoundError: If FAISS index directory is not found.
            DocumentPortalException: If retriever loading fails.
        """
        try:
            if not os.path.isdir(index_path):
                raise FileNotFoundError(f"FAISS index directory not found: {index_path}")

            embeddings = ModelLoader().load_embeddings()
            vectorstore = FAISS.load_local(
                index_path,
                embeddings,
                index_name=index_name,
                allow_dangerous_deserialization=True,
            )

            if search_kwargs is None:
                search_kwargs = {"k": k}

            self.retriever = vectorstore.as_retriever(
                search_type=search_type,
                search_kwargs=search_kwargs,
            )

            self._build_lcel_chain()

            self.log.info(
                "FAISS retriever loaded successfully",
                index_path=index_path,
                index_name=index_name,
                k=k,
                session_id=self.session_id,
            )

            return self.retriever
        except Exception as e:
            self.log.error(f"Failed to load Retriever from FAISS: {e}", error=str(e))
            raise DocumentPortalException("Failed to load Retriever from FAISS", sys)

    def invoke(self, user_input: str, chat_history: Optional[List[BaseMessage]] = None):
        """
        Invoke the RAG pipeline for a user query.

        Args:
            user_input (str): User's question or input text.
            chat_history (Optional[List[BaseMessage]]): Previous chat messages
                for context (default is empty list).

        Returns:
            str: Generated answer from the LLM.

        Raises:
            DocumentPortalException: If RAG chain is not initialized or invocation fails.
        """
        try:
            if self.chain is None:
                raise DocumentPortalException(
                    "RAG chain not initialized yet, Call load_retriever_from_fails instead",
                    sys
                )
            chat_history = chat_history or []
            payload = {"input": user_input, "chat_history": chat_history}
            answer = self.chain.invoke(payload)

            if not answer:
                self.log.warning(
                    "No answer generated", user_input=user_input, session_id=self.session_id
                )
                return "No answer by the model"
            self.log.info(
                "Chain invoked successfully, answer on the way",
                session_id=self.session_id,
                user_input=user_input,
                answer_preview=answer[:210],
            )
            return answer
        except Exception as e:
            self.log.error(f"Failed to invoke ConversationalRAG: {e}", error=str(e))
            raise DocumentPortalException("Failed to invoke ConversationalRAG", sys)

    def _load_llm(self):
        """
        Load the Large Language Model (LLM).

        Returns:
            llm: Loaded LLM instance.

        Raises:
            DocumentPortalException: If loading fails.
        """
        try:
            llm = ModelLoader().load_llm()
            if not llm:
                raise ValueError("LLM could not be loaded")
            self.log.info("LLM loaded Successfully", session_id=self.session_id)
            return llm
        except Exception as e:
            self.log.error(f"Failed to load LLM: {e}", error=str(e))
            raise DocumentPortalException("Failed to load LLM", sys)

    @staticmethod
    def _format_docs(docs) -> str:
        """
        Format retrieved documents into a single string.

        Args:
            docs (List[Document]): List of documents from retriever.

        Returns:
            str: Concatenated string of document page contents.
        """
        return "\n\n".join(getattr(d, "page_content", str(d)) for d in docs)

    def _build_lcel_chain(self):
        """
        Build the LangChain Expression Language (LCEL) graph.

        This pipeline includes:
            1. Rewriting user question with chat history.
            2. Retrieving relevant documents from retriever.
            3. Answer generation using retrieved context, user input, and chat history.

        Raises:
            DocumentPortalException: If retriever is not set or building fails.
        """
        try:
            if self.retriever is None:
                raise DocumentPortalException("No retriever set before building chain", sys)

            # 1) Rewrite user question with chat history context
            question_rewriter = (
                    {'input': itemgetter("input"), "chat_history": itemgetter("chat_history")}
                    | self.contextualize_prompt
                    | self.llm
                    | StrOutputParser()
            )

            # 2) Retrieve docs for rewritten question
            retrieve_docs = question_rewriter | self.retriever | self._format_docs

            # 3) Answer using retrieved context + original input + chat history
            self.chain = (
                    {
                        "context": retrieve_docs,
                        "input": itemgetter("input"),
                        "chat_history": itemgetter("chat_history"),
                    }
                    | self.qa_prompt
                    | self.llm
                    | StrOutputParser()
            )

            self.log.info("LCEL graph built successfully", session_id=self.session_id)

        except Exception as e:
            self.log.error("Failed to build LCEL chain", error=str(e))
            raise DocumentPortalException("Failed to build LCEL chain", sys)
