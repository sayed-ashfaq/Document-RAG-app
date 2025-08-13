import sys
import os
from dotenv import load_dotenv
from langchain_core.chat_history import BaseChatMessageHistory
from langchain_community.chat_message_histories import ChatMessageHistory
from langchain_community.vectorstores import FAISS
from langchain_core.runnables.history import RunnableWithMessageHistory
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain.chains import create_history_aware_retriever, create_retrieval_chain
import streamlit as st
from utils.model_loader import ModelLoader
from logger.custom_logger import CustomLogger
from exception.custom_exception import CustomException
from prompts.prompt_library import PROMPT_REGISTRY
from model.models import PromptType

load_dotenv()

class ConversationalRAG:
    def __init__(self, session_id: str, retriever)->None:
        self.log= CustomLogger().get_logger(__name__)
        self.session_id= session_id
        self.retriever= retriever
        try:
            self.llm= self._load_llm()
            self.contextualize_prompt= PROMPT_REGISTRY[PromptType.CONTEXTUALIZE_QUESTION.value]
            self.qa_prompt= PROMPT_REGISTRY[PromptType.CONTEXT_QA.value]

            self.history_aware_retriever= create_history_aware_retriever(
                self.llm, self.retriever, self.contextualize_prompt
            )
            self.log.info("Created history-aware retriever", retriever= str(retriever))

            self.qa_chain= create_stuff_documents_chain(self.llm, self.qa_prompt)
            self.rag_chain= create_retrieval_chain(self.history_aware_retriever, self.qa_chain)
            self.log.info("Created RAG chain", retriever= str(retriever))

            self.chain= RunnableWithMessageHistory(
                self.rag_chain,
                self._get_session_history,
                input_messages_key="input",
                history_messages_key= "chat_history",
                output_messages_key= "answer"
            )

            self.log.info("Wrapped chain with message history", session_id= self.session_id)


        except Exception as e:
            self.log.error("Failed to create conversational RAG", error= str(e))
            raise CustomException("Failed to create conversational RAG", sys)


    def _load_llm(self):
        try:
            llm= ModelLoader().load_llm()
            self.log.info("LLM loaded successfully",class_name= llm.__class__.__name__)
            return llm
        except Exception as e:
            self.log.error("Failed to load llm", error= str(e))
            raise CustomException("Failed to load llm", sys)

    def _get_session_history(self, session_id:str)->BaseChatMessageHistory:
        try:
            if "store" not in st.session_state:
                st.session_state.store= {}

            if session_id not in st.session_state.store:
                st.session_state.store[session_id]= ChatMessageHistory()
                self.log.info("New Chat session history created", session_id= session_id)
            return st.session_state.store[session_id]
        except Exception as e:
            self.log.error("Failed to get session history", error= str(e))
            raise CustomException("Failed to get session history", sys)

    def load_retriever_from_faiss(self, index_path:str):
        try:
            embeddings= ModelLoader().load_embedding()
            if not os.path.isdir(index_path):
                raise FileNotFoundError(f"FAISS index directory not found: {index_path}")

            vector_store= FAISS.load_local(index_path, embeddings)
            self.log.info("Loaded local FAISS index", index_path= index_path)
            return vector_store.as_retriever(search_type= "similarity", search_kwargs= {"k":5})
        except Exception as e:
            self.log.error("Failed to load retriever from faiss", error= str(e))
            raise CustomException("Failed to load retriever from faiss", sys)

    def invoke(self, user_input: str) -> str:
        try:
            response= self.chain.invoke(
                {"input": user_input},
                config= {"configurable": {"session_id": self.session_id}}
            )
            answer= response.get("answer", "No answer.")

            if not answer:
                self.log.warning("Empty answer received", session_id= self.session_id)

            self.log.info("CHAIN invoked successfully", session_id= self.session_id)
            return answer
        except Exception as e:
            self.log.error("Failed to invoke Conversational RAG", error= str(e))
            raise CustomException("Failed to invoke RAG chain", sys)