from pydantic import BaseModel, RootModel
from typing import List, Union
from enum import Enum

class Metadata(BaseModel):
    """
    Schema for document metadata.

    Attributes:
        Summary (List[str]): List of key highlights or points from the document.
        Title (str): Title of the document.
        Author (List[str]): List of authors.
        DateCreated (str): Original creation date of the document.
        LastModifiedDate (str): Last updated or modified date.
        Publisher (str): Name of the publisher.
        Language (str): Language of the document.
        PageCount (Union[int, str]): Number of pages (can also be "Not Available").
        SentimentTone (str): Overall tone or sentiment of the document.
    """
    Summary: List[str]
    Title: str
    Author: List[str]
    DateCreated: str
    LastModifiedDate: str
    Publisher: str
    Language: str
    PageCount: Union[int, str]
    SentimentTone: str


class ChangeFormat(BaseModel):
    """
    Schema for capturing changes in a document.

    Attributes:
        Page (str): Page number where the change occurred.
        Changes (str): Description of what was changed.
    """
    Page: str
    Changes: str


class SummaryResponse(RootModel[list[ChangeFormat]]):
    """
    Root schema for summarizing a list of document changes.

    The root model is a list of `ChangeFormat` objects.
    Example use case: returning all changes made during document comparison.
    """
    pass


class PromptType(str, Enum):
    """
    Enum for supported prompt types in document workflows.
    """
    DOCUMENT_ANALYSIS = "document_analysis"
    DOCUMENT_COMPARISON = "document_comparison"
    CONTEXTUALIZE_QUESTION = "contextualize_question"
    CONTEXT_QA = "context_qa"
