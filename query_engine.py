import os
import json
from typing import List, Dict, Any, Optional, Union
from dataclasses import dataclass

from langchain_community.embeddings import OpenAIEmbeddings, HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
from langchain_community.chat_models import ChatOpenAI
from langchain_community.llms import HuggingFacePipeline  # Add this import
from langchain.prompts import PromptTemplate
from langchain.chains import RetrievalQA, LLMChain
from langchain.docstore.document import Document
from langchain.chains.question_answering import load_qa_chain

import warnings
warnings.filterwarnings('ignore')

from utils.config import OPENAI_API_KEY, VECTOR_STORE_PATH, DEFAULT_LLM_MODEL, EMBEDDING_MODEL
from utils.math_parser import simplify_equation

@dataclass
class QueryResult:
    """Holds the result of a query"""
    answer: str
    sources: List[Dict[str, Any]]
    paper_ids: List[str]
    related_sections: List[str]
    confidence: float
    grounding: Dict[str, Any]

class QueryEngine:
    def __init__(self):
        """Initialize the query engine with LLM and vector store"""
        self.setup_embeddings()
        self.setup_llm()
        self.load_vector_store()
    
    def setup_embeddings(self):
        """Initialize embedding model based on environment"""
        try:
            if OPENAI_API_KEY:
                self.embeddings = OpenAIEmbeddings()
            else:
                self.embeddings = HuggingFaceEmbeddings(model_name=EMBEDDING_MODEL)
        except Exception as e:
            print(f"Error setting up embeddings: {e}")
            # Fallback to default sentence transformers model
            print("Falling back to default sentence-transformers model")
            self.embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
    
    def setup_llm(self):
        """Initialize LLM based on environment"""
        self.use_cloud_llm = bool(OPENAI_API_KEY)
        
        if self.use_cloud_llm:
            try:
                self.llm = ChatOpenAI(model=DEFAULT_LLM_MODEL, temperature=0.1)
                return
            except Exception as e:
                print(f"Error setting up OpenAI: {e}")
        
        # Try to use a lightweight Hugging Face model
        try:
            print("Loading HuggingFace model...")
            from transformers import pipeline
            # Use tiny model for simple text-generation
            tiny_llm = pipeline('text-generation', model='distilgpt2', max_new_tokens=256)
            self.llm = HuggingFacePipeline(pipeline=tiny_llm)
            print("Successfully loaded HuggingFace model")
            return
        except Exception as e:
            print(f"Error loading HuggingFace model: {e}")
            
        # If we reach here, no LLM is available - create a dummy LLM that returns error messages
        class DummyLLM:
            def __call__(self, *args, **kwargs):
                return {
                    "output_text": "Sorry, no LLM is available. Please set up OPENAI_API_KEY in your environment."
                }
                
        print("\n\n")
        print("=" * 80)
        print("ERROR: No LLM available!")
        print("To fix this, you need to either:")
        print("1. Set the OPENAI_API_KEY environment variable")
        print("   Example: export OPENAI_API_KEY=your-key-here")
        print("\nOR")
        print("2. Install required packages for local LLM:")
        print("   pip install transformers torch")
        print("=" * 80)
        print("\n\n")
        
        self.llm = DummyLLM()
    
    def load_vector_store(self):
        """Load the FAISS vector store"""
        if os.path.exists(VECTOR_STORE_PATH):
            self.vector_store = FAISS.load_local(
                VECTOR_STORE_PATH, 
                self.embeddings,
                allow_dangerous_deserialization=True  # Add this flag
            )
        else:
            raise FileNotFoundError(f"Vector store not found at {VECTOR_STORE_PATH}. Please ingest papers first.")
    
    def answer_query(
        self, 
        query: str, 
        paper_ids: Optional[List[str]] = None,
        k: int = 5,
        mode: str = "normal"
    ) -> QueryResult:
        """
        Run RAG over vectorstore and return an answer
        
        Args:
            query: The user question
            paper_ids: Optional list of paper IDs to restrict search to
            k: Number of chunks to retrieve
            mode: Query mode (normal, compare, equation)
            
        Returns:
            QueryResult with answer and sources
        """
        # Special handling for equation mode
        if mode == "equation":
            return self._handle_equation_query(query)
        
        # Filter by paper ID if specified
        if paper_ids:
            # Convert to filter function for FAISS
            def filter_func(doc):
                return doc.metadata.get("id") in paper_ids
            retriever = self.vector_store.as_retriever(search_kwargs={"k": k, "filter": filter_func})
        else:
            retriever = self.vector_store.as_retriever(search_kwargs={"k": k})
        
        # Retrieve relevant documents
        docs = retriever.get_relevant_documents(query)
        
        if not docs:
            return QueryResult(
                answer="I couldn't find any relevant information to answer your question.",
                sources=[],
                paper_ids=[],
                related_sections=[],
                confidence=0.0,
                grounding={}
            )
        
        # Extract metadata for tracking sources
        sources = []
        paper_ids = []
        sections = []
        
        for doc in docs:
            paper_id = doc.metadata.get("id")
            title = doc.metadata.get("title", "Unknown")
            source = {"id": paper_id, "title": title, "chunk": doc.page_content[:100] + "..."}
            
            if "section" in doc.metadata:
                source["section"] = doc.metadata["section"]
                sections.append(f"{doc.metadata['section']}")
            
            if paper_id not in paper_ids:
                paper_ids.append(paper_id)
            
            sources.append(source)
        
        # Build prompt based on mode
        if mode == "compare" and len(set(paper_ids)) > 1:
            prompt = self._build_comparison_prompt()
        else:
            prompt = self._build_standard_prompt()
        
        # Create QA chain
        qa_chain = load_qa_chain(
            llm=self.llm,
            chain_type="stuff",
            prompt=prompt
        )
        
        # Run the chain
        result = qa_chain({"input_documents": docs, "question": query})
        
        # Extract confidence and section awareness
        confidence = 0.7  # Default confidence - adjust if model provides confidence scores
        grounding = self._analyze_grounding(result["output_text"], sources)
        
        return QueryResult(
            answer=result["output_text"],
            sources=sources,
            paper_ids=paper_ids,
            related_sections=list(set(sections)),
            confidence=confidence,
            grounding=grounding
        )
    
    def _handle_equation_query(self, equation: str) -> QueryResult:
        """
        Handle equation simplification queries
        """
        try:
            simplified = simplify_equation(equation)
            return QueryResult(
                answer=f"Simplified equation: \n{simplified}",
                sources=[],
                paper_ids=[],
                related_sections=["Equation Simplification"],
                confidence=0.9,
                grounding={"equation": equation, "simplified": simplified}
            )
        except Exception as e:
            return QueryResult(
                answer=f"Error simplifying equation: {str(e)}",
                sources=[],
                paper_ids=[],
                related_sections=[],
                confidence=0.0,
                grounding={}
            )
    
    def _build_standard_prompt(self) -> PromptTemplate:
        """
        Build prompt template for standard queries
        """
        template = """You are SmartPaperQ, an AI research assistant that provides accurate, 
        helpful answers to questions about research papers.

        Answer the question based only on the following context:
        {context}

        Question: {question}

        Important instructions:
        1. Only answer based on information in the provided context
        2. If the context doesn't contain the answer, say "I don't have enough information to answer that"
        3. Cite specific sections when possible (e.g., "According to the Methods section...")
        4. Be concise but thorough in your explanations
        5. If mentioning numerical results, include actual values from the papers

        Answer:"""
        
        return PromptTemplate(
            template=template,
            input_variables=["context", "question"]
        )
    
    def _build_comparison_prompt(self) -> PromptTemplate:
        """
        Build prompt template for comparison queries
        """
        template = """You are SmartPaperQ, an AI research assistant that provides accurate, 
        helpful comparisons between research papers.

        Compare the papers described in the following context to answer the question:
        {context}

        Question: {question}

        Important instructions:
        1. Focus on comparing and contrasting the papers
        2. Structure your answer with clear comparison points
        3. Cite specific sections and results from each paper
        4. Point out similarities and differences in methods, results, and conclusions
        5. If the papers use different metrics, note that in your comparison
        6. Be objective and evidence-based in your comparison

        Comparison:"""
        
        return PromptTemplate(
            template=template,
            input_variables=["context", "question"]
        )
    
    def _analyze_grounding(self, answer: str, sources: List[Dict[str, Any]]) -> Dict[str, Any]:
        """
        Analyze the answer for grounding in sources
        
        Args:
            answer: The generated answer
            sources: List of source documents
            
        Returns:
            Grounding assessment
        """
        # Simple overlap-based grounding assessment
        grounding = {"score": 0.0, "sections": []}
        
        # Collect unique sections
        sections = set()
        for source in sources:
            if "section" in source:
                sections.add(source["section"])
        
        grounding["sections"] = list(sections)
        
        # Calculate rough grounding score based on source overlap
        # This is very simplistic - in a real system, you'd use a more sophisticated approach
        words = answer.lower().split()
        overlap_count = 0
        
        for source in sources:
            source_words = source["chunk"].lower().split()
            for word in words:
                if word in source_words and len(word) > 4:  # Only count substantial words
                    overlap_count += 1
        
        grounding["score"] = min(1.0, overlap_count / (len(words) + 0.1))
        
        return grounding

def answer_query(
    query: str, 
    paper_ids: Optional[List[str]] = None,
    mode: str = "normal"
) -> Dict[str, Any]:
    """
    High-level function to answer a query
    
    Args:
        query: User question
        paper_ids: Optional paper IDs to restrict search to
        mode: Query mode (normal, compare, equation)
        
    Returns:
        Dictionary with answer and metadata
    """
    engine = QueryEngine()
    result = engine.answer_query(query, paper_ids, mode=mode)
    
    # Convert to dict for API response
    return {
        "answer": result.answer,
        "sources": result.sources,
        "papers": result.paper_ids,
        "sections": result.related_sections,
        "confidence": result.confidence
    }
