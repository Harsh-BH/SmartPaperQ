import PyPDF2
import re
from typing import Dict, Any, Tuple, List
import nltk
import os
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

class PaperComparison:
    """Service for comparing scientific papers."""
    
    def __init__(self):
        # Set a specific download directory for NLTK data
        nltk_data_dir = os.path.join(os.path.expanduser('~'), 'nltk_data')
        os.makedirs(nltk_data_dir, exist_ok=True)
        
        # Download required NLTK resources directly without checking
        try:
            print(f"Downloading NLTK data to {nltk_data_dir}")
            nltk.download('punkt', download_dir=nltk_data_dir, quiet=False)
            nltk.download('stopwords', download_dir=nltk_data_dir, quiet=False)
            
            # Point NLTK to our custom data directory
            nltk.data.path.insert(0, nltk_data_dir)
        except Exception as e:
            print(f"Warning: Failed to download NLTK data: {e}")
            print("The application will try to continue, but may fail if resources are missing.")
    
    def extract_text_from_pdf(self, pdf_path: str) -> str:
        """Extract text content from a PDF file."""
        text = ""
        with open(pdf_path, 'rb') as file:
            pdf_reader = PyPDF2.PdfReader(file)
            for page_num in range(len(pdf_reader.pages)):
                text += pdf_reader.pages[page_num].extract_text()
        return text
    
    def preprocess_text(self, text: str) -> str:
        """Clean and preprocess text for comparison."""
        # Convert to lowercase
        text = text.lower()
        # Remove special characters and numbers
        text = re.sub(r'[^\w\s]', '', text)
        text = re.sub(r'\d+', '', text)
        
        try:
            # Tokenize
            tokens = word_tokenize(text)
            # Remove stopwords - with fallback if stopwords not available
            try:
                stop_words = set(stopwords.words('english'))
                filtered_tokens = [word for word in tokens if word not in stop_words]
            except LookupError:
                print("Warning: Stopwords not available. Continuing without stopword removal.")
                filtered_tokens = tokens
        except LookupError:
            print("Warning: Tokenizer not available. Falling back to basic splitting.")
            # Fallback to basic tokenization
            tokens = text.split()
            filtered_tokens = tokens
            
        return ' '.join(filtered_tokens)
    
    def compare_papers(self, pdf_path1: str, pdf_path2: str) -> Dict[str, Any]:
        """Compare two papers and return similarity metrics."""
        # Extract text from PDFs
        text1 = self.extract_text_from_pdf(pdf_path1)
        text2 = self.extract_text_from_pdf(pdf_path2)
        
        # Preprocess texts
        processed_text1 = self.preprocess_text(text1)
        processed_text2 = self.preprocess_text(text2)
        
        # Calculate TF-IDF and cosine similarity
        vectorizer = TfidfVectorizer()
        tfidf_matrix = vectorizer.fit_transform([processed_text1, processed_text2])
        similarity = cosine_similarity(tfidf_matrix[0:1], tfidf_matrix[1:2])[0][0]
        
        # Get common keywords
        feature_names = vectorizer.get_feature_names_out()
        dense1 = tfidf_matrix[0].todense().tolist()[0]
        dense2 = tfidf_matrix[1].todense().tolist()[0]
        
        keywords1 = {feature_names[i]: dense1[i] for i in range(len(feature_names)) if dense1[i] > 0.1}
        keywords2 = {feature_names[i]: dense2[i] for i in range(len(feature_names)) if dense2[i] > 0.1}
        
        common_keywords = set(keywords1.keys()) & set(keywords2.keys())
        
        return {
            'similarity_score': similarity,
            'common_keywords': list(common_keywords),
            'paper1_length': len(text1),
            'paper2_length': len(text2),
        }
