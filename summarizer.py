from typing import List, Dict, Any, Optional
import os

from langchain_community.chat_models import ChatOpenAI
from langchain_community.llms import CTransformers, OpenAI
from langchain.prompts import PromptTemplate
from langchain.chains import LLMChain
from langchain_community.vectorstores import FAISS
from langchain_community.embeddings import OpenAIEmbeddings, HuggingFaceEmbeddings
from langchain.chains.summarize import load_summarize_chain
from langchain.schema import Document

from utils.config import OPENAI_API_KEY, VECTOR_STORE_PATH, DEFAULT_LLM_MODEL, EMBEDDING_MODEL

class PaperSummarizer:
    """Generate various types of summaries for research papers"""
    
    def __init__(self):
        """Initialize summarizer with LLM and vector store"""
        self.setup_embeddings()
        self.setup_llm()
        self.load_vector_store()
    
    def setup_embeddings(self):
        """Setup embedding model based on environment"""
        if OPENAI_API_KEY:
            self.embeddings = OpenAIEmbeddings()
        else:
            self.embeddings = HuggingFaceEmbeddings(model_name=EMBEDDING_MODEL)
    
    def setup_llm(self):
        """Setup LLM based on environment"""
        if OPENAI_API_KEY:
            self.llm = ChatOpenAI(model=DEFAULT_LLM_MODEL, temperature=0.2)
        else:
            # Try to load a local model
            try:
                self.llm = CTransformers(
                    model="llama-2-7b-chat.ggmlv3.q4_0.bin",
                    model_type="llama",
                    config={"temperature": 0.2, "max_new_tokens": 1000}
                )
            except Exception as e:
                print(f"Error loading local model: {e}")
                print("Falling back to smaller local model")
                try:
                    self.llm = CTransformers(
                        model="ggml-model-q4_0.bin",  # Adjust as needed
                        model_type="gpt2",
                        config={"temperature": 0.2, "max_new_tokens": 500}
                    )
                except Exception as e2:
                    print(f"Error loading fallback model: {e2}")
                    print("No LLM available. Please set OPENAI_API_KEY or download a local model.")
                    raise RuntimeError("No LLM available")
    
    def load_vector_store(self):
        """Load FAISS vector store"""
        if os.path.exists(VECTOR_STORE_PATH):
            self.vector_store = FAISS.load_local(
                VECTOR_STORE_PATH, 
                self.embeddings,
                allow_dangerous_deserialization=True  # Add this flag
            )
        else:
            raise FileNotFoundError(f"Vector store not found at {VECTOR_STORE_PATH}. Please ingest papers first.")
    
    def get_paper_chunks(self, paper_id: str) -> List[Document]:
        """
        Retrieve all chunks for a specific paper
        
        Args:
            paper_id: The paper ID to retrieve
            
        Returns:
            List of Document objects containing paper content
        """
        # Create filter function for FAISS
        def filter_func(doc):
            return doc.metadata.get("id") == paper_id
            
        # Get all document chunks for this paper
        docs = self.vector_store.similarity_search(
            "abstract introduction methods results",  # Generic query to get main sections
            k=25,  # Get enough chunks to have full paper
            filter=filter_func
        )
        
        # Sort chunks by their position in the paper
        docs.sort(key=lambda doc: doc.metadata.get("chunk_id", 0))
        
        return docs
    
    def generate_technical_summary(self, paper_id: str) -> str:
        """
        Generate a technical summary of the paper
        
        Args:
            paper_id: Paper ID to summarize
            
        Returns:
            Technical summary string
        """
        # Get paper content
        docs = self.get_paper_chunks(paper_id)
        
        if not docs:
            return "Paper not found or contains no content."
        
        # Create the summary chain
        prompt_template = """
        Generate a comprehensive technical summary of this research paper for an expert audience.
        Focus on the key technical contributions, methodologies, algorithms, and results.
        Use precise technical terminology appropriate for the field.
        Include quantitative results with specific metrics when available.
        Highlight the technical innovation and positioning within the research area.
        
        Paper content:
        {text}
        
        Technical Summary:
        """
        
        prompt = PromptTemplate(template=prompt_template, input_variables=["text"])
        
        # Create a summarize chain
        chain = load_summarize_chain(
            self.llm,
            chain_type="map_reduce",
            map_prompt=prompt,
            combine_prompt=prompt,
            verbose=False
        )
        
        # Run the chain
        result = chain.run(docs)
        
        return result
    
    def generate_layman_summary(self, paper_id: str) -> str:
        """
        Generate an accessible summary for non-experts
        
        Args:
            paper_id: Paper ID to summarize
            
        Returns:
            Layman-friendly summary string
        """
        # Get paper content
        docs = self.get_paper_chunks(paper_id)
        
        if not docs:
            return "Paper not found or contains no content."
        
        # Create the summary chain
        prompt_template = """
        Create an accessible summary of this research paper for a non-expert audience.
        Explain the key concepts in plain language, avoiding jargon when possible.
        When technical terms are necessary, explain them briefly.
        Focus on the problem being solved, why it matters, and the main findings.
        Use analogies or real-world examples where appropriate to aid understanding.
        
        Paper content:
        {text}
        
        Plain English Summary:
        """
        
        prompt = PromptTemplate(template=prompt_template, input_variables=["text"])
        
        # Create a summarize chain
        chain = load_summarize_chain(
            self.llm,
            chain_type="map_reduce",
            map_prompt=prompt,
            combine_prompt=prompt,
            verbose=False
        )
        
        # Run the chain
        result = chain.run(docs)
        
        return result
    
    def generate_section_summary(self, paper_id: str, section_type: str) -> str:
        """
        Generate a summary of a specific section type
        
        Args:
            paper_id: Paper ID to summarize
            section_type: Type of section to focus on (methods, results, etc.)
            
        Returns:
            Section summary string
        """
        # Get paper content
        docs = self.get_paper_chunks(paper_id)
        
        if not docs:
            return "Paper not found or contains no content."
        
        # Filter for section types if metadata is available
        section_docs = []
        section_keywords = {
            "methods": ["method", "approach", "methodology", "algorithm", "implementation"],
            "results": ["result", "evaluation", "experiment", "performance", "finding"],
            "key_findings": ["contribution", "finding", "conclusion", "result", "implication"]
        }
        
        keywords = section_keywords.get(section_type.lower(), [section_type.lower()])
        
        for doc in docs:
            # Check section metadata if available
            if "section" in doc.metadata:
                section_name = doc.metadata["section"].lower()
                if any(k in section_name for k in keywords):
                    section_docs.append(doc)
            
            # Also check content for section headers
            content_lower = doc.page_content.lower()
            if any(f"{k}:" in content_lower or f"{k}\n" in content_lower for k in keywords):
                if doc not in section_docs:
                    section_docs.append(doc)
        
        # If no matching sections found, use whole paper
        if not section_docs:
            section_docs = docs
        
        # Create appropriate template based on section type
        prompts = {
            "methods": """
                Summarize the methodology of this research paper.
                Focus on the approaches, algorithms, models, datasets, and experimental setup.
                Highlight any novel techniques or unique implementation details.
                
                Paper content:
                {text}
                
                Methods Summary:
            """,
            "results": """
                Summarize the key results and findings from this research paper.
                Include important metrics, measurements, and performance evaluations.
                Highlight how the results compare to previous or baseline approaches.
                Mention any ablation studies or additional analyses.
                
                Paper content:
                {text}
                
                Results Summary:
            """,
            "key_findings": """
                Extract and summarize the key findings and contributions of this research paper.
                Focus on the most important discoveries, innovations, or insights.
                Highlight the significance of these findings for the field.
                Include any limitations or future work mentioned.
                
                Paper content:
                {text}
                
                Key Findings:
            """
        }
        
        prompt_template = prompts.get(section_type.lower(), f"""
            Summarize the {section_type} section of this research paper.
            
            Paper content:
            {{text}}
            
            {section_type.title()} Summary:
        """)
        
        prompt = PromptTemplate(template=prompt_template, input_variables=["text"])
        
        # Create a summarize chain
        chain = load_summarize_chain(
            self.llm,
            chain_type="map_reduce",
            map_prompt=prompt,
            combine_prompt=prompt,
            verbose=False
        )
        
        # Run the chain
        result = chain.run(section_docs)
        
        return result
    
    def compare_papers(self, paper_ids: List[str], aspect: str) -> str:
        """
        Generate comparative analysis of multiple papers
        
        Args:
            paper_ids: List of paper IDs to compare
            aspect: Aspect to compare (overall, methodology, results, strengths_weaknesses)
            
        Returns:
            Comparative analysis string
        """
        if len(paper_ids) < 2:
            return "Need at least two papers to compare."
        
        # Collect paper contents
        paper_contents = {}
        paper_titles = {}
        
        for paper_id in paper_ids:
            docs = self.get_paper_chunks(paper_id)
            
            if not docs:
                return f"Paper {paper_id} not found or contains no content."
            
            # Extract title from metadata
            title = docs[0].metadata.get("title", f"Paper {paper_id}")
            paper_titles[paper_id] = title
            
            # Combine all chunks into one text
            paper_contents[paper_id] = "\n\n".join([doc.page_content for doc in docs])
        
        # Create the comparison prompt based on aspect
        aspect_prompts = {
            "overall_approach": """
                Compare these research papers, focusing on their overall approaches:
                
                {paper_contents}
                
                In your comparison:
                1. Identify the main problem each paper addresses
                2. Compare their high-level approaches and methodologies
                3. Contrast their key assumptions and perspectives
                4. Compare their theoretical foundations
                5. Highlight conceptual similarities and differences
                
                Overall Approach Comparison:
            """,
            "methodology_differences": """
                Compare the methodologies of these research papers:
                
                {paper_contents}
                
                In your comparison:
                1. Contrast the specific techniques and algorithms used
                2. Compare the experimental setups and conditions
                3. Highlight differences in datasets or data preparation
                4. Compare evaluation metrics and methodology
                5. Note any differences in implementation details or parameters
                
                Methodology Comparison:
            """,
            "key_results": """
                Compare the key results of these research papers:
                
                {paper_contents}
                
                In your comparison:
                1. Summarize the main findings from each paper
                2. Compare performance metrics and quantitative results
                3. Contrast statistical significance and confidence levels
                4. Compare how each paper interprets their results
                5. Highlight where results agree or conflict between papers
                
                Results Comparison:
            """,
            "strengths_weaknesses": """
                Compare the strengths and weaknesses of these research papers:
                
                {paper_contents}
                
                In your comparison:
                1. Identify key strengths of each paper
                2. Note limitations or weaknesses of each approach
                3. Compare the thoroughness of evaluations
                4. Contrast how well each paper addresses potential criticisms
                5. Assess which approach might be more suitable in different scenarios
                
                Strengths & Weaknesses Comparison:
            """
        }
        
        # Fallback to general comparison if aspect not found
        prompt_template = aspect_prompts.get(aspect.lower(), """
            Compare these research papers:
            
            {paper_contents}
            
            In your comparison:
            1. Summarize each paper's main approach and contributions
            2. Compare their methodologies and results
            3. Highlight key similarities and differences
            4. Discuss relative strengths and limitations
            
            Paper Comparison:
        """)
        
        # Format the papers with titles for input
        formatted_papers = []
        for paper_id, content in paper_contents.items():
            title = paper_titles[paper_id]
            # Trim content if too large
            trim_content = content[:5000] + "..." if len(content) > 5000 else content
            formatted_papers.append(f"PAPER: {title}\n\n{trim_content}\n\n")
        
        # Join all papers
        all_papers = "\n" + "-" * 40 + "\n".join(formatted_papers)
        
        # Create and run the chain
        prompt = PromptTemplate(template=prompt_template, input_variables=["paper_contents"])
        chain = LLMChain(llm=self.llm, prompt=prompt)
        result = chain.run(paper_contents=all_papers)
        
        return result

def generate_summary(paper_id: str, mode: str = "technical_tldr") -> str:
    """
    Generate a summary of a paper
    
    Args:
        paper_id: ID of the paper to summarize
        mode: Type of summary to generate (technical_tldr, layman, key_findings, methods, results)
        
    Returns:
        Summary text
    """
    summarizer = PaperSummarizer()
    
    if mode == "technical_tldr":
        return summarizer.generate_technical_summary(paper_id)
    elif mode == "laymans_summary":
        return summarizer.generate_layman_summary(paper_id)
    elif mode in ["key_findings", "methods_summary", "results_analysis"]:
        section_type = mode.replace("_summary", "").replace("_analysis", "")
        return summarizer.generate_section_summary(paper_id, section_type)
    else:
        # Default to technical summary
        return summarizer.generate_technical_summary(paper_id)

def compare_papers(paper_ids: List[str], aspect: str = "overall_approach") -> str:
    """
    Compare multiple papers
    
    Args:
        paper_ids: List of paper IDs to compare
        aspect: Aspect to compare on (overall_approach, methodology_differences, 
                key_results, strengths_weaknesses)
        
    Returns:
        Comparative analysis
    """
    summarizer = PaperSummarizer()
    return summarizer.compare_papers(paper_ids, aspect)
