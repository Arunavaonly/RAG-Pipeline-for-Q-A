# Added Features:
1. Memory component added with ConversationBufferMemory
2. Chat window enabled within Streamlit UI
3. All mathematical formulations are now rendered using Latex

# NCERT Tutor: Math and Science Q&A

This project is a Streamlit-based application that acts as a tutor for NCERT Mathematics and Science textbooks. It leverages AI models to provide simple, child-friendly explanations to questions from these textbooks.

## Features

- **Interactive Q&A**: Ask questions from NCERT Math and Science textbooks and get detailed, easy-to-understand answers.
- **AI-Powered**: Uses LangChain, FAISS, and Google Generative AI embeddings for document retrieval and question answering.
- **Customizable UI**: Includes a user-friendly interface with enhanced design elements.
- **Document Similarity Search**: Displays relevant document excerpts for better context.

## Prerequisites

- Python 3.8 or higher
- Virtual environment (recommended)
- API keys for:
  - Groq API (`GROQ_API_KEY`)
  - Google Generative AI (`GOOGLE_API_KEY`)

## Installation

1. Clone the repository:
   ```bash
   git clone <repository-url>
   cd RAG-Pipeline-for-Q-A
   ```

2. Set up a virtual environment:
   ```bash
   python -m venv venv
   source venv/bin/activate  # On Windows: venv\Scripts\activate
   ```

3. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```

4. Create a `.env` file in the project root and add your API keys:
   ```
   GROQ_API_KEY=your_groq_api_key
   GOOGLE_API_KEY=your_google_api_key
   ```

5. Add NCERT textbook PDFs:
   - Place Math textbooks in the `Mathematics` folder.
   - Place Science textbooks in the `Science` folder.

## Usage

1. Run the application:
   ```bash
   streamlit run app.py
   ```

2. Open the app in your browser (default: `http://localhost:8501`).

3. Use the sidebar to load data:
   - Click **Load Math Data** to process Math textbooks.
   - Click **Load Science Data** to process Science textbooks.

4. Ask questions in the respective sections for Math and Science:
   - Enter your question in the input box.
   - Click **Get Answer** to retrieve the response.

5. View document similarity results under the **Document Similarity Search** expander.

## Project Structure

- `app.py`: Main application file.
- `requirements.txt`: Python dependencies.
- `.gitignore`: Files and folders to ignore in version control.
- `Mathematics/`: Folder for Math textbooks.
- `Science/`: Folder for Science textbooks.

## Technologies Used

- **Streamlit**: For building the web interface.
- **LangChain**: For creating document retrieval and question-answering chains.
- **FAISS**: For vector-based document search.
- **Google Generative AI**: For embeddings and semantic chunking.

## Future Enhancements

- Add support for more subjects.
- Improve document similarity search results.
- Optimize performance for large datasets.