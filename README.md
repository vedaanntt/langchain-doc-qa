# LangChain Document Q&A System 📄🤖 (🚧 Work in Progress)

A Python-based Retrieval-Augmented Generation (RAG) pipeline designed to extract, process, and ultimately answer questions based on PDF documents. Currently in active development, the foundational document loading and smart-chunking modules have been implemented using LangChain, setting the stage for future vector storage and LLM integration.

## Features
* **PDF Document Parsing (Completed):** Leverages `PyPDFLoader` to seamlessly ingest multi-page PDF documents.
* **Smart Text Chunking (Completed):** Uses `RecursiveCharacterTextSplitter` to break down large texts into digestible, 1000-character chunks with strategic overlap to preserve contextual continuity.
* **Vector Embeddings (Planned):** Future integration with a vector database to store and retrieve document chunks.
* **LLM Question Answering (Planned):** Upcoming functionality to pass retrieved context to a Large Language Model for precise answers.

## Tech Stack
* **Language:** [Python 3](https://www.python.org/)
* **Framework:** [LangChain](https://www.langchain.com/)
* **Document Processing:** PyPDF

## Prerequisites
Before you begin, ensure you have the following installed on your machine:
* **Python 3.x:** Download and install from [python.org](https://www.python.org/downloads/).
* **Sample Data:** Place a test PDF file named `sample.pdf` in the root directory for the script to process.

## Requirements
To run this project, you will need the following Python packages installed. 
<pre>
langchain-community
langchain-text-splitters
pypdf
</pre>


## Installation & Setup
Follow these steps to get the project running on your local machine:

1. **Clone the repository:** Download the project files to your computer using Git.
2. **Navigate to the directory:** Change your terminal's current directory to the newly cloned project folder.
3. **Create a virtual environment (Recommended):** This keeps the project's dependencies isolated from your system's global Python packages.
4. **Activate the environment:** Turn on the virtual environment so that packages are installed locally. *(Note: The command differs slightly for Windows).*
5. **Install dependencies:** Use `pip` to install LangChain and the necessary document loaders.

## Directory structure:
<pre>
langchain-doc-qa/
├── qa_system.py      
└── README.md
</pre>
