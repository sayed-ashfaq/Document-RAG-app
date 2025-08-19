import os
import sys
from dotenv import load_dotenv
from utils.config_loader import load_config
from langchain_google_genai import GoogleGenerativeAIEmbeddings
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_groq import ChatGroq
# from langchain_openai import ChatOpenAI
from logger.custom_logger import CustomLogger
from exception.custom_exception_archive import DocumentPortalException

# create logger
log = CustomLogger().get_logger(__name__)

class ModelLoader:
    """
    A helper class to load environment variables, configuration,
    embeddings, and LLM (Large Language Models) based on settings.
    """

    def __init__(self):
        """
        Initializes the ModelLoader:
        - Loads environment variables from `.env`
        - Validates required API keys
        - Loads configuration from `config.yaml`
        """
        load_dotenv()  # load environment variables from .env file
        self._validate_env()
        self.config = load_config()
        log.info("Configuration loaded Successfully", config_keys=list(self.config.keys()))

    def _validate_env(self):
        """
        Validates that required environment variables are set.
        Raises a custom exception if any are missing.
        """
        required_vars = ["GOOGLE_API_KEY", "GROQ_API_KEY"]

        # Collect all API keys into dictionary
        self.api_key = {key: os.getenv(key) for key in required_vars}

        # Check which ones are missing
        missing = [k for k, v in self.api_key.items() if not v]

        if missing:
            log.error("Missing required environment variables: {}".format(missing))
            raise DocumentPortalException("Missing required environment variables: {}".format(missing), sys)

        log.info("Environment variables validated",
                 available_key=[k for k in self.api_key.keys() if self.api_key[k]])

    def load_embeddings(self):
        """
        Loads and returns the Google Generative AI Embedding model
        (used to convert text into vector embeddings).
        """
        try:
            log.info("Loading Google Generative AI embeddings...")
            model_name = self.config["embedding_model"]["model_name"]
            return GoogleGenerativeAIEmbeddings(model=model_name)
        except Exception as e:
            log.error("Failed to load Google Generative AI embeddings: {}".format(e))
            raise DocumentPortalException("Fail to load embedding model", sys)

    def load_llm(self, model_name="google"):
        """
        Loads and returns an LLM (Large Language Model) based on provider (Google or Groq).

        Args:
            model_name (str): The key for which LLM provider to load (default: "google")

        Returns:
            LLM object (ChatGoogleGenerativeAI or ChatGroq)
        """
        # get config block for llm
        llm_block = self.config["llm"]
        log.info("Loading LMM models...")

        # check if provider is available in config.yaml
        provider_key = model_name
        if provider_key not in llm_block:
            log.error("LLM provider not found in config", provider_key=provider_key)
            raise ValueError(f"LLM provider not found in config: {provider_key}")

        # read llm config values
        llm_config = llm_block[provider_key]
        provider = llm_config.get("provider")
        model_name = llm_config.get("model_name")
        temperature = llm_config.get("temperature")
        max_tokens = llm_config.get("max_tokens")

        log.info("Loading LMM models...", provider=provider, model_name=model_name,
                 temperature=temperature, max_tokens=max_tokens)

        # choose provider
        if provider == "groq":
            # load groq llm
            llm = ChatGroq(
                model=model_name,
                api_key=self.api_key["GROQ_API_KEY"],
                temperature=temperature,
            )
            return llm

        elif provider == "google":
            # load google llm
            llm = ChatGoogleGenerativeAI(
                model=model_name,
                temperature=temperature,
                google_api_key=self.api_key["GOOGLE_API_KEY"],
            )
            return llm

        else:
            log.error("Unknown provider", provider=provider)
            raise ValueError(f"Unknown provider: {provider}")


if __name__ == "__main__":
    # create loader
    loader = ModelLoader()

    # Test embeddings model
    embedding_model = loader.load_embeddings()
    print(f"Embedding model loaded: {embedding_model}")

    # test embedding by converting text into vectors
    result = embedding_model.embed_query("Hello World")
    print(f"Embedding Result: {result}")

    # Test the LLM loading based on YAML config
    llm = loader.load_llm()
    print(f"LMM model loaded: {llm}")

    # test the model by invoking it with a simple query
    result = llm.invoke("Hello World")
    print(f"LLM Result: {result.content}")
