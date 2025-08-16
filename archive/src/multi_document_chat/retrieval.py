import sys
import os
from dotenv import load_dotenv
from operator import itemgetter
from langchain_core.output_parsers import StrOutputParser
from typing import Optional, List

from langchain_core.messages import BaseMessage
from langchain_community.vectorstores import FAISS

from utils.model_loader import ModelLoader
from logger.custom_logger import CustomLogger
from exception.custom_exception_archive import DocumentPortalException
from prompts.prompt_library import PROMPT_REGISTRY
from model.models import PromptType
from langchain_core.prompts import ChatPromptTemplate

load_dotenv()

class ConversationalRAG:
    def __init__(self, session_id: str, retriever= None):
        try:
            self.log= CustomLogger().get_logger(__name__)
            self.session_id= session_id
            self.llm= self._load_llm()
            self.contextualize_prompt: ChatPromptTemplate= PROMPT_REGISTRY[PromptType.CONTEXTUALIZE_QUESTION.value]
            self.qa_prompt: ChatPromptTemplate= PROMPT_REGISTRY[PromptType.CONTEXT_QA.value]

            if retriever is None:
                raise ValueError("retriever cannot be done")

            self.retriever= retriever
            self._build_lcel_chain()
            self.log.info("ConversationalRAG initialized", session_id= self.session_id)


        except Exception as e:
            self.log.error("Failed to initialize contextualize_prompt", error= str(e))
            raise DocumentPortalException("Failed to initialize contextualize_prompt", sys)

    def load_retriever_from_faiss(self, index_path: str):
        """
        Load a FAISS vectorstore from disk and convert to retriever
        :return:
        """
        try:
            embeddings= ModelLoader().load_embedding()
            if not os.path.isdir(index_path):
                raise FileNotFoundError(f"FAISS index directory not found: {index_path}")
            vectorstore= FAISS.load_local(
                index_path=index_path,
                embeddings=embeddings,
                allow_dangerous_deserialization=True
            )

            self.retriever= vectorstore.as_retriever(search_type= "similarity", search_kwargs= {"k": 5})
            self.log.info("Retriever successfully loaded", session_id= self.session_id)
            return self.retriever
        except Exception as e:
            self.log.error("Failed to load retriever from faiss", error= str(e))
            raise DocumentPortalException("Failed to load retriever from faiss", sys)

    def invoke(self, user_input: str, chat_history: Optional[List[BaseMessage]]= None) -> str:
        """
        package of chain
        :param user_input:
        :param chat_history:
        :return:
        """
        try:
            chat_history = chat_history or []
            payload= {"input": user_input, "chat_history": chat_history or []}
            answer = self.chain.invoke(payload)
            if not answer:
                self.log.warning("No answer has been generated", session_id= self.session_id)
                return "No answer generated"

            self.log.info("Chain invoke successfully",
                          session_id= self.session_id,
                          user_input= user_input,
                          answer_preview= answer[:100])
            return answer

        except Exception as e:
            self.log.error("Failed to invoke retriever", error= str(e))
            raise DocumentPortalException("Failed to invoke retriever", sys)

    def _load_llm(self):
        try:
            llm= ModelLoader().load_llm()
            if not llm:
                raise ValueError("LLM could not be loaded")
            return llm
        except Exception as e:
            self.log.error('Failed to load LLM', error= str(e))
            raise DocumentPortalException("Failed to load LLM", sys)

    @staticmethod
    def _format_docs(docs):
        return "\n\n".join(d.page_content for d in docs)

    def _build_lcel_chain(self):
        """
        Combining two chains
        :return:
        """
        try:

            question_rewriter = (
                {
                    "input": itemgetter("input"), "chat_history": itemgetter("chat_history")
                }
                | self.contextualize_prompt
                | self.llm
                | StrOutputParser()
            )
            retrieved_docs= question_rewriter | self.retriever | self._format_docs


            self.chain = (
                {
                    "context": retrieved_docs,
                    "input": itemgetter("input"),
                    "chat_history" : itemgetter("chat_history"),
                }
                | self.qa_prompt
                | self.llm
                |StrOutputParser()
            )

            self.log.info("chain has been built", session_id= self.session_id)
        except Exception as e:
            self.log.error("Failed to build LCEL chain", error= str(e))
            raise DocumentPortalException("Failed to build LCEL chain", sys)

