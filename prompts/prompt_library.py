# Prepare prompt template
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder

document_analysis = ChatPromptTemplate.from_template(
    """
    You are a highly capable assistant trained to analyze and summarize documents.
    Return ONLY valid JSON matching the exact schema below.
    
    {format_instructions}
    
    Analyze this document:
    {document_text}
    """
)


document_comparison= ChatPromptTemplate.from_template(
    """
    You will be provided with content from two PDFs. Your tasks are as follows:
    1. Compare the content in two PDFs
    2. Identify the difference in PDF and not down the page number
    3. The output you provide must be page wise comparison content.
    4. If any page do not have any change, mention as "NO Change Found".
    
    Input documents: 
    {combined_docs}
    
    output format:
    {format_instructions}
    """
)

contextualize_question_prompt  = ChatPromptTemplate.from_messages([
    ("system", (
        "Given a conversation history and the most recent user query, rewrite the query as a standalone question "
        "that makes sense without relying on the previous context. Do not provide an answer—only reformulate the "
        "question if necessary; otherwise, return it unchanged."
    )),
    MessagesPlaceholder("chat_history"),
    ("human", "{input}"),

])

context_qa_prompt = ChatPromptTemplate.from_messages([
    ("system", (
"You are an assistant designed to answer questions using the provided context. Rely only on the retrieved "
        "information to form your response. If the answer is not found in the context, respond with 'I don't know.' "
        "Keep your answer concise and no longer than three sentences.\n\n{context}"
    )),
    MessagesPlaceholder("chat_history"),
    ("human", "{input}"),
])


PROMPT_REGISTRY = {
    "document_analysis": document_analysis,
    "document_comparison": document_comparison,
    "contextualize_question": contextualize_question_prompt ,
    "context_qa": context_qa_prompt,
}
