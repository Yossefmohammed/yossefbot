<<<<<<< HEAD
# Search in Your PDF

A powerful PDF question-answering application built with Streamlit and LangChain that allows you to ask questions about your PDF documents and get AI-powered responses.

## Features

- Upload and process PDF documents
- Ask questions about your PDF content
- Get AI-powered responses using LaMini-T5 model
- Modern and user-friendly interface
- Persistent vector storage using Chroma

## Prerequisites

- Python 3.8 or higher
- CUDA-capable GPU (recommended for better performance)
- Git

## Installation

1. Clone the repository:
```bash
git clone https://github.com/yourusername/Search-in-Your-PDF.git
cd Search-in-Your-PDF
```

2. Create a virtual environment (recommended):
```bash
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
```

3. Install the required packages:
```bash
pip install streamlit langchain-community sentence-transformers chromadb transformers torch
```

4. Download the required models:

### LaMini-T5 Model
```bash
# Create a directory for the model
mkdir -p models/LaMini-T5-738M

# Download the model from Hugging Face
python -c "from transformers import AutoTokenizer, AutoModelForSeq2SeqLM; AutoTokenizer.from_pretrained('MBZUAI/LaMini-T5-738M').save_pretrained('models/LaMini-T5-738M'); AutoModelForSeq2SeqLM.from_pretrained('MBZUAI/LaMini-T5-738M').save_pretrained('models/LaMini-T5-738M')"
```

### Sentence Transformer Model
The sentence transformer model (all-miniLM-L6-v2) will be automatically downloaded when you first run the application.

## Database Setup and Embedding Process

1. Create the database directory:
```bash
mkdir -p chroma_db
```

2. The application uses ChromaDB for storing document embeddings and vectors. The database will be automatically created when you first process a PDF document. The process includes:

   - Document chunking: PDFs are split into smaller chunks for better processing
   - Embedding generation: Each chunk is converted into vector embeddings using the all-miniLM-L6-v2 model
   - Vector storage: Embeddings are stored in ChromaDB for efficient similarity search
   - Persistence: The database is automatically persisted to disk in the `chroma_db` directory

3. The database structure:
   - `chroma_db/`: Main directory for the vector database
   - `chroma_db/chroma.sqlite3`: SQLite database file storing metadata
   - `chroma_db/embeddings/`: Directory containing the vector embeddings

4. To reset the database (if needed):
```bash
rm -rf chroma_db/*
```

## Usage

1. Start the application:
```bash
streamlit run app.py
```

2. Open your web browser and navigate to the URL shown in the terminal (usually http://localhost:8501)

3. Upload your PDF documents and start asking questions!

## Project Structure

- `app.py`: Main application file
- `constant.py`: Configuration settings
- `models/`: Directory containing the downloaded models
- `chroma_db/`: Directory for persistent vector storage
  - `chroma.sqlite3`: Database file
  - `embeddings/`: Vector embeddings storage

## Technical Details

- Uses LaMini-T5-738M for text generation
- Employs all-miniLM-L6-v2 for text embeddings
- ChromaDB for vector storage
- Streamlit for the web interface

## Contributing

Contributions are welcome! Please feel free to submit a Pull Request.

## License

This project is licensed under the MIT License - see the LICENSE file for details.
=======
# yossefbot
>>>>>>> c43d2e93ad837a66434df210fcab9e768c0b2924
# YossefBot

AI chatbot project using LangChain and RAG.
