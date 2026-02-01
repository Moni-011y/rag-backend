import os
from typing import List, Tuple
from langchain_community.vectorstores import FAISS
from langchain_groq import ChatGroq
from langchain_google_genai import GoogleGenerativeAIEmbeddings
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.documents import Document
from langchain_classic.memory import ConversationBufferMemory
from langchain_classic.chains import ConversationalRetrievalChain

class RAGEngine:
    def __init__(self, index_dir: str):
        self.index_dir = index_dir
        print("Initializing Google Gemini Embeddings API...")
        self.embeddings = GoogleGenerativeAIEmbeddings(model="models/embedding-001")
        self.vector_store = None
        self.indexed_files = set()
        self.last_uploaded_file = None
        self.memories = {}  # session_id -> memory object
        print(f"RAG Engine initialized with index_dir: {index_dir}")
        
        # Load existing index if it exists
        if os.path.exists(os.path.join(index_dir, "index.faiss")):
            self.vector_store = FAISS.load_local(index_dir, self.embeddings, allow_dangerous_deserialization=True)
            # We would need to store/load metadata about indexed_files separately for a persistent state
            # For this demo, we'll keep it in memory.

    def add_documents(self, chunks: List[str], filename: str):
        print(f"Adding {len(chunks)} documents from {filename} to FAISS...")
        docs = [Document(page_content=chunk, metadata={"source": filename}) for chunk in chunks]
        
        if self.vector_store is None:
            print("Creating new FAISS index...")
            self.vector_store = FAISS.from_documents(docs, self.embeddings)
        else:
            print("Updating existing FAISS index...")
            self.vector_store.add_documents(docs)
        
        self.indexed_files.add(filename)
        self.last_uploaded_file = filename
        # self.vector_store.save_local(self.index_dir)

    def get_indexed_files(self) -> List[str]:
        return list(self.indexed_files)

    def remove_document(self, filename: str) -> bool:
        if filename in self.indexed_files:
            self.indexed_files.remove(filename)
            if self.last_uploaded_file == filename:
                self.last_uploaded_file = None
                # If there are other files, we could pick the most recent, but for now we reset.
            # Note: FAISS doesn't support easy deletion by metadata. 
            # In a production app, we would clear and rebuild or use a different vector store.
            return True
        return False

    def query(self, question: str, session_id: str) -> Tuple[str, List[str]]:
        print(f"Querying for: {question}")
        if self.vector_store is None:
            print("Error: No documents indexed.")
            return "No documents indexed yet. Please upload some files.", []

        if session_id not in self.memories:
            print(f"Initializing new memory for session: {session_id}")
            self.memories[session_id] = ConversationBufferMemory(
                memory_key="chat_history", return_messages=True, output_key='answer'
            )

        print("Initializing Groq LLM (llama-3.3-70b-versatile)...")
        llm = ChatGroq(model_name="llama-3.3-70b-versatile", temperature=0.3)
        
        print("Setting up Retrieval Chain...")
        
        # Configure retriever with metadata filter if a document was uploaded
        search_kwargs = {"k": 5}
        if self.last_uploaded_file:
            print(f"Limiting search context to the latest document: {self.last_uploaded_file}")
            search_kwargs["filter"] = {"source": self.last_uploaded_file}
        else:
            print("Warning: No specific document targeted (no 'last_uploaded_file' set).")

        chain = ConversationalRetrievalChain.from_llm(
            llm=llm,
            retriever=self.vector_store.as_retriever(search_kwargs=search_kwargs),
            memory=self.memories[session_id],
            return_source_documents=True
        )

        print("Executing chain...")
        try:
            result = chain.invoke({"question": question})
            answer = result["answer"]
            # Strictly return only the last uploaded file as the source to match user requirement
            sources = [self.last_uploaded_file] if self.last_uploaded_file else []
            print(f"Returned strict source: {sources}")
            return answer, sources
        except Exception as e:
            print(f"Chain execution failed: {str(e)}")
            raise e
