# 🧠 SmartPaperQ — AI-Powered Research Paper Assistant

SmartPaperQ is a next-gen **Retrieval-Augmented Generation (RAG)** application designed to assist researchers in navigating and understanding complex research papers. Powered by **Groq's ultra-fast LLM API**, it provides accurate and contextually relevant answers to your research queries with minimal latency.

## Features

- **Intelligent Query Processing**: Understands and processes complex research questions
- **Contextual Answers**: Provides answers based on the context of the research paper
- **Ultra-Fast Inference**: Uses Groq API for lightning-fast responses (10-100x faster than typical APIs)
- **User-Friendly Interface**: Easy-to-use interface for seamless interaction
- **Multi-Document Support**: Handles multiple research papers simultaneously
- **Smart Caching**: Optional document caching for faster repeated uploads
- **Multiple Models**: Choice between Llama 3 (70B & 8B) and Mixtral models

## Why Groq?

SmartPaperQ leverages Groq's LPU (Language Processing Unit) Inference API for:
- **Ultra-low latency**: Generate answers in a fraction of the time compared to other APIs
- **High throughput**: Process more queries with less waiting
- **Cost efficiency**: Groq provides competitive pricing for LLM inference
- **Top-tier models**: Access to state-of-the-art models like Llama 3 70B and Mixtral

## Installation

To install SmartPaperQ, follow these steps:

1. Clone the repository:
   ```bash
   git clone https://github.com/Harsh-BH/Paper-shaper.git
   ```
2. Navigate to the project directory:
   ```bash
   cd Paper-shaper
   ```
3. Install the required dependencies:
   ```bash
   pip install -r requirements.txt
   ```
4. Create a `.env` file in the root directory and add your Groq API key:
   ```
   GROQ_API_KEY=your_groq_api_key
   ```

To obtain a Groq API key:
1. Visit [console.groq.com](https://console.groq.com) and sign up for an account
2. Navigate to API Keys section and create a new key
3. Copy the key to your `.env` file

## Usage

To start using SmartPaperQ, run the following command:
```bash
streamlit run app.py
```

Or use the convenience script:
```bash
python run.py
```

Then open your browser and navigate to `http://localhost:8000` to access the web interface.

### Quick Start Guide

1. **Upload Paper**: Use the sidebar to upload your research paper in PDF format
2. **Process Document**: Click "Process Paper" to extract and index the document content
3. **Select Model**: Choose from available LLM models based on your needs:
   - Llama 3 70B: Fastest with excellent quality (recommended)
   - Llama 3 8B: Balanced speed and quality
   - Mixtral 8x7B: Largest context window for complex papers
4. **Ask Questions**: Enter your research questions in the main area
5. **View Answers**: Get AI-generated answers with source references

## Project Structure

```
Paper-shaper/
├── app.py                # Main application entry point
├── requirements.txt      # Project dependencies
├── .env                  # Environment variables (create this file)
├── README.md             # Project documentation
├── LICENSE               # MIT license
├── cache/                # Document cache directory (created at runtime)
└── src/                  # Source code
    ├── components/       # Core components
    │   ├── document_processor.py
    │   ├── embeddings.py
    │   ├── retriever.py
    │   └── generator.py
    ├── models/           # Model definitions
    │   └── rag_pipeline.py
    ├── interface/        # User interface
    │   └── web_app.py
    ├── utils/            # Utility functions
    │   └── helpers.py
    └── config/           # Configuration
        └── settings.py
```

## Performance Benchmarks

SmartPaperQ with Groq delivers exceptional performance:

| Operation | Average Time |
|-----------|--------------|
| Document Processing (10 pages) | ~3-5 seconds |
| Query Response (Llama 3 70B) | ~1-2 seconds |
| Query Response (Mixtral 8x7B) | ~2-4 seconds |

*Times may vary based on document complexity and system resources*

## Contributing

We welcome contributions to SmartPaperQ! If you have any ideas, suggestions, or bug reports, please open an issue or submit a pull request.

## License

This project is licensed under the MIT License. See the [LICENSE](LICENSE) file for details.
<<<<<<< HEAD
=======

## Contact

For any questions or inquiries, please contact us at [email@example.com](mailto:email@example.com).
<<<<<<< HEAD
>>>>>>> 3bc5255f369d3d778d8c0b3379466e68791b3a6c
=======
>>>>>>> 3bc5255f369d3d778d8c0b3379466e68791b3a6c
