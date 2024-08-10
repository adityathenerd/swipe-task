from langchain_community.document_loaders import PyPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_google_genai import GoogleGenerativeAIEmbeddings
from langchain.vectorstores import FAISS
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain_core.prompts import ChatPromptTemplate
from langchain.chains import create_retrieval_chain
from langchain_google_genai import ChatGoogleGenerativeAI
import os
from dotenv import load_dotenv

load_dotenv()

def main(file_path: str):

    def loader_func(file_path: str):
        loader = PyPDFLoader(file_path=file_path)
        documents = loader.load()

        text = RecursiveCharacterTextSplitter(
        chunk_size=700,  # The maximum number of characters in a chunk: we selected this value arbitrarily
        chunk_overlap=100,  # The number of characters to overlap between chunks
        add_start_index=True,  # If `True`, includes chunk's start index in metadata
        strip_whitespace=True,  # If `True`, strips whitespace from the start and end of every document
        ).split_documents(documents)

        return text

    text = loader_func(file_path=file_path)
    embeddings = GoogleGenerativeAIEmbeddings(model = "models/embedding-001", google_api_key=os.getenv("google_api_key"))

    vectorstore = FAISS.from_documents(text, embeddings)
    vectorstore.save_local("vectorstore.db")

    retriever = vectorstore.as_retriever()

    llm = ChatGoogleGenerativeAI(model="gemini-pro", google_api_key=os.getenv("google_api_key"))

    template = """
    You are an assistant for question-answering tasks.
    also clean the text from any placeholders or '\n's
    Use the provided context only to answer the following question:
    also clean the text from any placeholders or '\n's
    <context>
    {context}
    </context>

    Question: {input}
    """

    # Create a prompt template
    prompt = ChatPromptTemplate.from_template(template)

    # Create a chain 
    doc_chain = create_stuff_documents_chain(llm, prompt)
    chain = create_retrieval_chain(retriever, doc_chain)

    def rag_answer(example: dict):
        response = chain.invoke({"input": (example["question"])})
        return response["answer"]

    return rag_answer({
        "question": '''
                        a) Give me complete details of costumer in the invoice.
                        b) Mention the products being sold.
                        c) Mention the total amount spent.
                        
                        Output a Dictionary in the following format having answer to questions as asked before
                        {
                            "Costumer Details": "",
                            "Products": "",
                            "Total Amount": ""
                        
                        }
                    '''
    })