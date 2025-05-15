import fitz  # PyMuPDF
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain.memory import ConversationBufferMemory
from langchain.chains import ConversationalRetrievalChain
from langchain.prompts import PromptTemplate

# Constants
GEMINI_MODEL_ID = "gemini-1.5-flash-latest"
DEFAULT_K_CONTEXT_CHUNKS = 4
DEFAULT_TEMPERATURE = 0.5

def load_embeddings_model():
    return HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")

def extract_text_from_pdf(pdf_file_bytes):
    try:
        doc = fitz.open(stream=pdf_file_bytes, filetype="pdf")
        text = "\n".join([page.get_text() for page in doc])
        doc.close()
        return text if text.strip() else None
    except Exception as e:
        raise RuntimeError(f"Error reading PDF: {e}")

def create_vector_store(pdf_text):
    try:
        splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
        chunks = splitter.split_text(pdf_text)
        embeddings = load_embeddings_model()
        return FAISS.from_texts(texts=chunks, embedding=embeddings)
    except Exception as e:
        raise RuntimeError(f"Error creating vector store: {e}")

def conversation_answering(vector_store, question, api_key,
                           k=DEFAULT_K_CONTEXT_CHUNKS,
                           temperature=DEFAULT_TEMPERATURE,
                           conversation_obj=None):
    if not api_key or not vector_store:
        return {"answer": "Missing API key or vector store.", "source_documents": []}, conversation_obj

    try:
        if conversation_obj is None:
            # Prompt Templates
            qa_template = """You are an expert AI research assistant...
Context:
---
{context}
---

Question: {question}
Precise Answer:"""
            QA_PROMPT = PromptTemplate(template=qa_template, input_variables=["context", "question"])

            condense_prompt = PromptTemplate.from_template("""Given the chat and a follow-up question...
Chat History:
{chat_history}
Follow Up Input: {question}
Standalone question:""")

            # Langchain Setup
            llm = ChatGoogleGenerativeAI(
                model=GEMINI_MODEL_ID,
                google_api_key=api_key,
                temperature=temperature,
                convert_system_message_to_human=True
            )
            retriever = vector_store.as_retriever(search_kwargs={'k': k})
            memory = ConversationBufferMemory(
                memory_key='chat_history',
                output_key='answer',
                return_messages=True
            )
            chain = ConversationalRetrievalChain.from_llm(
                llm=llm,
                retriever=retriever,
                memory=memory,
                condense_question_prompt=condense_prompt,
                return_source_documents=True,
                combine_docs_chain_kwargs={"prompt": QA_PROMPT},
                output_key='answer'
            )
            conversation_obj = {"chain": chain, "memory": memory}

        result = conversation_obj["chain"].invoke({'question': question})
        return result, conversation_obj

    except Exception as e:
        return {"answer": f"Error during processing: {e}", "source_documents": []}, conversation_obj
