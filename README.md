# Document-based Question Answering Application

## Overview
This application allows users to upload PDF documents, process them into text chunks, and ask questions about the content. The system uses vector embeddings for document retrieval via **Pinecone**, and generates answers using **Cohere's** language model. The interface is powered by **Streamlit**, making it simple and interactive.

## Features
- **Document Upload**: Supports PDF documents for text extraction.
- **Conversational QA**: Users can ask questions about the document, and the system will respond with contextually relevant answers.
- **Multi-document support**: Process multiple documents at once and ask questions across them.
- **Memory**: Retains conversational history for follow-up questions based on prior interactions.

## Tech Stack
- **Frontend**: [Streamlit](https://streamlit.io/)
- **Backend**: [LangChain](https://github.com/hwchase17/langchain), [Pinecone](https://www.pinecone.io/), [Cohere API](https://cohere.ai/), [HuggingFace Embeddings](https://huggingface.co/)
- **Vector Database**: [Pinecone](https://www.pinecone.io/)

## Setup Instructions

### Prerequisites
- Python 3.8 or above.
- **Pinecone** API key: [Sign up for Pinecone](https://www.pinecone.io/) and get an API key.
- **Cohere** API key: [Sign up for Cohere](https://cohere.com/) and get an API key.

### Installation

1. **Clone the repository**:
   ```bash
   git clone https://github.com/your-username/document-qa-app.git
   cd document-qa-app

2. **Create a virtual environment**:
    ```bash
    python -m venv env
    source env/bin/activate  # For Windows, use `env\Scripts\activate`

3. **Install dependencies**:
    ```bash
    pip install -r requirements.txt

4. **Set up environment variables**:
    - Create a **.env** file in the root directory and add the following keys
     ```bash
        PINECONE_API_KEY=your_pinecone_api_key
        COHERE_API_KEY=your_cohere_api_key

5. **Run the application**:
    ```bash
     streamlit run app.py


### Usage Instructions
#### 1.Uploading a Document:

-  Use the sidebar to upload one or more PDF documents.
-  Click the Process button to extract and process the documents.

#### 2.Asking Questions:

- Type your question in the input box at the top of the page.
- The system will retrieve relevant information from the uploaded document(s) and respond accordingly.


#### 3.Conversation History:

- The conversation history is displayed in the chat interface, allowing users to follow up on previous questions.
