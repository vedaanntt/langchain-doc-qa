from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter

def process_document(file_path):
    """take the pdf and split it into chunks."""
    """returns splitted chunks as a list"""
    
    print(f"Loading document: {file_path}")
    loader = PyPDFLoader(file_path)
    documents = loader.load()
    print(f"Successfully loaded {len(documents)} pages.")


    print("Chunking texts")
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=1000, # max characters per chunk
        chunk_overlap=200, # characters shared between chunks to prevent loss of context
        length_function=len
    )
    
    chunks = text_splitter.split_documents(documents)
    print(f"Document split into {len(chunks)} chunks.")
        
    return chunks


if __name__ == "__main__":
    chunks = process_document("sample.pdf")