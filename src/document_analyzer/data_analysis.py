import os
import sys
from utils.model_loader import ModelLoader
from logger.custom_logger import CustomLogger
from exception.custom_exception import DocumentPortalException
from model.models import *
from langchain_core.output_parsers import JsonOutputParser
from langchain.output_parsers import OutputFixingParser
from prompts.prompt_library import PROMPT_REGISTRY


class DocumentAnalyzer:
    """
    Handles document analysis using a pre-trained LLM.
    Supports metadata extraction, structured output, and session-based logging.
    """

    def __init__(self):
        """
        Initialize the DocumentAnalyzer with:
            - ModelLoader for loading LLMs
            - Parsers for structured JSON output
            - Predefined prompts from PROMPT_REGISTRY

        Raises:
            DocumentPortalException: If initialization fails.
        """
        self.log = CustomLogger().get_logger(__name__)
        try:
            self.loader = ModelLoader()
            self.llm = self.loader.load_llm(model_name="google")

            # Output parsers - ensures the LLM output comes back as structured JSON that matches your Metadata Pydantic model.
            self.parser = JsonOutputParser(pydantic_object=Metadata)

            # Output fixing parser - if the LLM output is invalid JSON, it re-asks the LLM to “fix” it into the correct format.
            self.fixing_parser = OutputFixingParser.from_llm(
                parser=self.parser,
                llm=self.llm
            )

            self.prompt = PROMPT_REGISTRY["document_analysis"]

            self.log.info("DocumentAnalyzer initialized successfully")

        except Exception as e:
            self.log.error(f"Error initializing DocumentAnalyzer: {e}")
            raise DocumentPortalException("Error in DocumentAnalyzer initialization", sys)

    def analyze_document(self, document_text: str) -> dict:
        """
        Extract metadata and structured summary from document text.

        Args:
            document_text (str): The raw text of the document.

        Returns:
            dict: Extracted metadata and analysis results
                  (follows the `Metadata` Pydantic schema).

        Raises:
            DocumentPortalException: If analysis fails.
        """
        try:
            # chain pipeline - Order matters in chain
            chain = self.prompt | self.llm | self.fixing_parser
            self.log.info("Meta-data analysis chain initialized")

            response = chain.invoke({
                "format_instructions": self.parser.get_format_instructions(),
                "document_text": document_text
            })

            self.log.info("Metadata extraction successful", keys=list(response.keys()))
            return response

        except Exception as e:
            self.log.error("Metadata analysis failed", error=str(e))
            raise DocumentPortalException("Metadata extraction failed") from e
