import sys

import pandas as pd
from dotenv import load_dotenv
from langchain.output_parsers import OutputFixingParser
from langchain_core.output_parsers import JsonOutputParser

from exception.custom_exception_archive import DocumentPortalException
from logger.custom_logger import CustomLogger
from model.models import SummaryResponse
from prompts.prompt_library import PROMPT_REGISTRY
from utils.model_loader import ModelLoader


class DocumentComparatorLLM:
    """
    Document Comparator powered by an LLM.

    This class provides functionality to compare two or more documents
    using a Large Language Model (LLM) with structured JSON output.
    The pipeline uses LangChain's prompt chaining, a JSON parser, and
    an output-fixing parser for robust parsing.

    Responsibilities:
        - Initialize and manage the LLM with required parsers.
        - Run document comparisons through the LLM.
        - Format and return structured comparison results as DataFrames.

    Attributes:
        log: Custom logger instance.
        loader: ModelLoader instance for loading LLM.
        llm: Large Language Model instance.
        parser: JSON output parser based on a Pydantic schema (SummaryResponse).
        fixing_parser: Output parser that attempts to fix malformed outputs.
        prompt: Document comparison prompt template.
        chain: LangChain pipeline for processing comparison requests.
    """

    def __init__(self):
        """
        Initialize DocumentComparatorLLM.

        Loads environment variables, initializes logger, loads the LLM,
        sets up parsers and chain for document comparison.

        Raises:
            DocumentPortalException: If initialization fails.
        """
        try:
            load_dotenv()
            self.log = CustomLogger().get_logger(__name__)
            self.loader = ModelLoader()
            self.llm = self.loader.load_llm()

            # Structured output parser with validation
            self.parser = JsonOutputParser(pydantic_object=SummaryResponse)
            self.fixing_parser = OutputFixingParser.from_llm(
                parser=self.parser, llm=self.llm
            )

            # Prompt chain
            self.prompt = PROMPT_REGISTRY["document_comparison"]
            self.chain = self.prompt | self.llm | self.parser

            self.log.info("DocumentComparatorLLM has been initialized")
        except Exception as e:
            self.log.error("Failed to initialize DocumentComparatorLLM", error=str(e))
            raise DocumentPortalException("Error initializing DocumentComparatorLLM", sys)

    def compare_documents(self, combined_docs: str) -> pd.DataFrame:
        """
        Compare two or more documents using the LLM.

        Args:
            combined_docs (str): Concatenated document contents to be compared.

        Returns:
            pd.DataFrame: Structured comparison results (rows = comparison units).

        Raises:
            DocumentPortalException: If LLM invocation or parsing fails.
        """
        try:
            inputs = {
                "combined_docs": combined_docs,
                "format_instructions": self.parser.get_format_instructions(),
            }

            self.log.info("Starting document comparison", inputs=inputs)
            response = self.chain.invoke(inputs)
            self.log.info(
                "Chain invoked successfully",
                response_preview=str(response)[:200]
            )

            return self._format_response(response)
        except Exception as e:
            self.log.error("Error in compare_documents", error=str(e))
            raise DocumentPortalException("Error in compare_documents", sys)

    def _format_response(self, response_parsed: list[dict]) -> pd.DataFrame:
        """
        Internal helper to format LLM response into a DataFrame.

        Args:
            response_parsed (list[dict]): Parsed response from LLM.

        Returns:
            pd.DataFrame: Comparison results as a structured dataframe.

        Raises:
            DocumentPortalException: If response formatting fails.
        """
        try:
            df = pd.DataFrame(response_parsed)
            self.log.info("Response formatted into dataframe", dataframe=df)
            return df
        except Exception as e:
            self.log.error("Error in _format_response", error=str(e))
            raise DocumentPortalException("Error in format_response", sys)
