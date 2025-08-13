import sys
from dotenv import load_dotenv
import pandas as pd
from logger.custom_logger import CustomLogger
from exception.custom_exception import CustomException
from model.models import *
from prompts.prompt_library import PROMPT_REGISTRY
from utils.model_loader import ModelLoader
from langchain_core.output_parsers import JsonOutputParser
from langchain.output_parsers import OutputFixingParser

class DocumentComparatorLLM:
    def __init__(self):
        load_dotenv()
        self.log= CustomLogger().get_logger(__name__)
        self.loader= ModelLoader()
        self.llm= self.loader.load_llm()
        self.parser= JsonOutputParser(pydantic_object=SummaryResponse)
        self.fixing_parser= OutputFixingParser.from_llm(parser= self.parser, llm= self.llm)
        self.prompt= PROMPT_REGISTRY["document_comparison"]
        self.chain= self.prompt | self.llm | self.parser
        self.log.info("DocumentComparatorLLM has been initialized")



    def compare_documents(self, combined_docs: str) -> pd.DataFrame:
        """
        Compare two documents and returns a structured comparison
        :return:
        """
        try:
            inputs= {
                "combined_docs": combined_docs,
                "format_instructions": self.parser.get_format_instructions(),
            }
            self.log.info(f"Document comparison completed", inputs= inputs)
            response= self.chain.invoke(inputs)
            self.log.info("Chain invoked successfully", response_preview= str(response)[:200])
            return self._format_response(response)
        except Exception as e:
            self.log.error(f"Error in compare documents: {e}")
            raise CustomException("Error in compare documents:", sys)


    def _format_response(self, response_parsed: list[dict]) -> pd.DataFrame: #underscore mean it's private not for public used in init
        """
        Formats the response from the LLM into structured format
        :return:
        """
        try:
            df= pd.DataFrame(response_parsed)
            self.log.info("Response formatted into dataframe", dataframe= df)
            return df
        except Exception as e:
            self.log.error(f"Error in format response: {e}")
            raise CustomException("Error in format response:", sys)