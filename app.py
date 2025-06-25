import streamlit as st
import os
import tempfile
from pathlib import Path
import pandas as pd
import time
from typing import List, Dict, Any
import base64
import uuid  # Add this import

from ingest import PaperIngestor
from query_engine import answer_query
from summarizer import generate_summary, compare_papers
from citation_graph import build_graph

# Page configuration
st.set_page_config(
    page_title="SmartPaperQ - Research Paper Assistant",
    page_icon="🧠",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Create needed directories
Path("./papers").mkdir(exist_ok=True)
Path("./vectorstore").mkdir(exist_ok=True)

# App state management
@st.cache_resource
def get_paper_ingestor():
    return PaperIngestor()

# Add this function to force refresh of the ingestor
def refresh_ingestor():
    # Generate a new key to force cache invalidation
    if 'ingestor_key' not in st.session_state:
        st.session_state.ingestor_key = str(uuid.uuid4())
    else:
        st.session_state.ingestor_key = str(uuid.uuid4())
    
    # Force recreation of ingestor
    st.cache_resource.clear()
    return get_paper_ingestor()

def get_papers() -> List[Dict[str, Any]]:
    """Get list of ingested papers"""
    try:
        # Make sure we have the latest ingestor
        ingestor = get_paper_ingestor()
        papers = []
        paper_ids_seen = set()  # Track papers we've already added
        
        # Debug
        print(f"Checking vectorstore for papers...")
        
        # Get metadata for papers in vectorstore
        for doc_id in ingestor.vector_store.docstore._dict.keys():
            doc = ingestor.vector_store.docstore._dict[doc_id]
            
            # Debug
            if doc_id.startswith("doc"):
                print(f"Document {doc_id} metadata: {doc.metadata}")
            
            paper_id = doc.metadata.get('id')
            
            # Skip if no paper ID or already processed this ID
            if not paper_id or paper_id in paper_ids_seen:
                continue
            
            # Add to our tracking set
            paper_ids_seen.add(paper_id)
            
            # Add paper to list
            paper_title = doc.metadata.get('title', 'Unknown Title')
            if paper_title == 'Unknown Title' and doc.page_content:
                # Try to extract title from content
                first_line = doc.page_content.strip().split('\n')[0][:50]
                paper_title = f"Document: {first_line}..."
                
            papers.append({
                'id': paper_id,
                'title': paper_title,
                'source': doc.metadata.get('source', 'Unknown'),
                'authors': doc.metadata.get('authors', [])
            })
        
        # Log how many papers we found for debugging
        print(f"Found {len(papers)} papers in the vectorstore")
        
        # If no papers found using our standard method, try an alternate approach
        if not papers:
            print("No papers found with standard method, trying alternate approach...")
            # Group by unique content signatures as a fallback
            content_sigs = {}
            for doc_id in ingestor.vector_store.docstore._dict.keys():
                doc = ingestor.vector_store.docstore._dict[doc_id]
                
                # Create a content signature from first 100 chars
                content_sig = doc.page_content[:100] if doc.page_content else ""
                if content_sig and content_sig not in content_sigs:
                    content_sigs[content_sig] = doc
            
            # Create papers from unique content signatures
            for i, (content_sig, doc) in enumerate(content_sigs.items()):
                if len(content_sig) < 10:  # Skip very short content
                    continue
                    
                # Generate title from content
                first_line = doc.page_content.strip().split('\n')[0][:50]
                paper_title = f"Document {i+1}: {first_line}..."
                
                papers.append({
                    'id': f"doc-{i}",
                    'title': paper_title,
                    'source': 'unknown',
                    'authors': []
                })
                
            print(f"Found {len(papers)} papers using alternate method")
            
            # If we found papers with alternate method, fix the vectorstore
            if papers:
                from fix_papers_metadata import fix_vectorstore_metadata
                fix_vectorstore_metadata()
                
        return papers
    except Exception as e:
        st.error(f"Error loading papers: {e}")
        import traceback
        traceback.print_exc()
        print(f"Exception in get_papers: {e}")
        return []

def display_html(html_content, height=500):
    """Display HTML content in Streamlit"""
    html_file = f"<iframe seamless style='width:100%;height:{height}px;border:none;' srcdoc='{html_content}'></iframe>"
    st.markdown(html_file, unsafe_allow_html=True)

def show_pdf(file_path):
    """Display PDF in Streamlit"""
    with open(file_path, "rb") as f:
        base64_pdf = base64.b64encode(f.read()).decode('utf-8')
    pdf_display = f'<iframe src="data:application/pdf;base64,{base64_pdf}" width="100%" height="600" type="application/pdf"></iframe>'
    st.markdown(pdf_display, unsafe_allow_html=True)

def main():
    # Sidebar navigation
    st.sidebar.title("🧠 SmartPaperQ")
    
    # Navigation options
    pages = {
        "Paper Q&A": "ask",
        "Upload Papers": "upload", 
        "arXiv Explorer": "arxiv",
        "Citation Graph": "graph",
        "Equation Simplifier": "equation",
        "Paper Summaries": "summary"
    }
    
    selection = st.sidebar.radio("Go to", list(pages.keys()))
    page = pages[selection]
    
    # Display papers list in sidebar
    with st.sidebar.expander("📚 Available Papers", expanded=True):
        papers = get_papers()
        if papers:
            paper_df = pd.DataFrame(papers)[['title', 'source']]
            st.dataframe(paper_df, use_container_width=True)
        else:
            st.info("No papers loaded. Upload some papers first!")
    
    # Bottom of sidebar - credits
    st.sidebar.markdown("---")
    st.sidebar.caption("Built with LangChain, FAISS & Streamlit")
    
    # Main content area based on selected page
    if page == "ask":
        render_qa_page()
    elif page == "upload":
        render_upload_page()
    elif page == "arxiv":
        render_arxiv_page()
    elif page == "graph":
        render_graph_page()
    elif page == "equation":
        render_equation_page()
    elif page == "summary":
        render_summary_page()

def render_qa_page():
    st.title("📝 Ask Questions About Papers")
    
    # Get available papers
    papers = get_papers()
    if not papers:
        st.warning("No papers available. Please upload or fetch papers first.")
        return
    
    # Paper selection
    paper_titles = {p['title']: p['id'] for p in papers}
    selected_titles = st.multiselect(
        "Select papers to query (optional - leave empty to search all papers)",
        options=list(paper_titles.keys())
    )
    
    selected_paper_ids = [paper_titles[title] for title in selected_titles] if selected_titles else None
    
    # Query mode selection
    query_modes = {
        "Standard Query": "normal",
        "Compare Papers": "compare"
    }
    
    mode_selection = st.radio("Query Mode", options=list(query_modes.keys()), horizontal=True)
    query_mode = query_modes[mode_selection]
    
    # If compare mode, enforce at least 2 papers
    if query_mode == "compare" and (not selected_paper_ids or len(selected_paper_ids) < 2):
        st.warning("Please select at least 2 papers to compare.")
    
    # Query input
    query = st.text_area("Enter your question about the paper(s)", height=100)
    
    if query and st.button("Submit Question", type="primary"):
        try:
            with st.spinner("Processing your question..."):
                # Get answer from RAG system
                result = answer_query(query, selected_paper_ids, mode=query_mode)
                
                # Display answer
                st.markdown("### Answer")
                st.markdown(result["answer"])
                
                # Display sources
                if result["sources"]:
                    with st.expander("View Sources", expanded=False):
                        for i, source in enumerate(result["sources"]):
                            st.markdown(f"**Source {i+1}:** {source['title']}")
                            if "section" in source:
                                st.markdown(f"*Section: {source['section']}*")
                            st.text(source["chunk"])
                            st.markdown("---")
                
                # Display confidence
                confidence = result.get("confidence", 0.0)
                confidence_color = "green" if confidence >= 0.7 else "orange" if confidence >= 0.4 else "red"
                st.markdown(f"<span style='color:{confidence_color}'>Confidence: {confidence:.2f}</span>", 
                           unsafe_allow_html=True)
        
        except Exception as e:
            st.error(f"Error processing question: {str(e)}")

def render_upload_page():
    st.title("📤 Upload Research Papers")
    
    # Direct file upload
    uploaded_file = st.file_uploader("Upload PDF research paper", type="pdf")
    
    if uploaded_file:
        try:
            # Save to temp file
            with tempfile.NamedTemporaryFile(delete=False, suffix='.pdf') as tmp_file:
                tmp_file.write(uploaded_file.getvalue())
                pdf_path = tmp_file.name
            
            # Show the PDF
            with st.expander("Preview PDF", expanded=True):
                show_pdf(pdf_path)
            
            # Process the PDF when requested
            if st.button("Process Paper", type="primary"):
                with st.spinner("Processing paper..."):
                    ingestor = get_paper_ingestor()
                    
                    # Create metadata
                    metadata = {
                        "title": uploaded_file.name.replace('.pdf', ''),
                        "source": "upload",
                        "file_path": pdf_path
                    }
                    
                    # Process the PDF
                    success = ingestor.process_pdf(pdf_path, metadata)
                    
                    if success:
                        st.success("Paper processed successfully!")
                        # Refresh the ingestor before rerunning to ensure new data is loaded
                        refresh_ingestor()
                        st.rerun()
                    else:
                        st.error("Failed to process paper.")
        except Exception as e:
            st.error(f"Error processing file: {str(e)}")
    
    # Process entire directory
    st.markdown("### Process Local Directory")
    dir_path = st.text_input("Path to directory containing PDFs")
    
    if dir_path and st.button("Process Directory"):
        if not os.path.isdir(dir_path):
            st.error(f"Directory not found: {dir_path}")
        else:
            with st.spinner("Processing directory..."):
                try:
                    ingestor = get_paper_ingestor()
                    processed = ingestor.ingest_directory(dir_path)
                    st.success(f"Successfully processed {len(processed)} papers from {dir_path}")
                    # Refresh the ingestor before rerunning
                    refresh_ingestor()
                    st.rerun()
                except Exception as e:
                    st.error(f"Error processing directory: {str(e)}")

def render_arxiv_page():
    st.title("🔍 arXiv Paper Explorer")
    
    # arXiv category selection
    categories = [
        "cs.AI", "cs.CL", "cs.CV", "cs.LG", "cs.NE", "cs.RO", 
        "stat.ML", "q-bio", "q-fin", "physics"
    ]
    
    selected_categories = st.multiselect(
        "Select arXiv categories",
        options=categories,
        default=["cs.CL", "cs.AI"]
    )
    
    # Number of papers to fetch
    max_papers = st.slider("Number of papers to fetch", min_value=1, max_value=30, value=5)
    
    if selected_categories and st.button("Fetch Recent Papers", type="primary"):
        with st.spinner(f"Fetching papers from arXiv ({', '.join(selected_categories)})..."):
            try:
                ingestor = get_paper_ingestor()
                papers = ingestor.fetch_arxiv_papers(selected_categories, max_papers)
                
                if not papers:
                    st.warning("No papers found matching your criteria.")
                    return
                
                # Display the papers
                st.success(f"Found {len(papers)} papers")
                
                # Create a dataframe for display
                paper_df = pd.DataFrame([{
                    "Title": p["title"],
                    "Authors": ", ".join(p["authors"][:2]) + ("..." if len(p["authors"]) > 2 else ""),
                    "ID": p["id"],
                    "Download": p["pdf_link"] is not None
                } for p in papers])
                
                st.dataframe(paper_df, use_container_width=True)
                
                # Option to download and process the papers
                if st.button("Download & Process Selected Papers"):
                    with st.spinner("Downloading and processing papers..."):
                        processed_count = 0
                        progress_bar = st.progress(0)
                        
                        for i, paper in enumerate(papers):
                            st.caption(f"Processing: {paper['title']}")
                            pdf_path = ingestor.download_arxiv_paper(paper)
                            
                            if pdf_path:
                                if ingestor.process_pdf(pdf_path, paper):
                                    processed_count += 1
                                
                                # Be nice to arXiv API
                                time.sleep(1)
                            
                            # Update progress
                            progress_bar.progress((i + 1) / len(papers))
                        
                        if processed_count > 0:
                            st.success(f"Successfully processed {processed_count} papers!")
                            # Refresh the ingestor before rerunning
                            refresh_ingestor()
                            st.rerun()
                        else:
                            st.error("Failed to process any papers.")
                            
            except Exception as e:
                st.error(f"Error fetching papers: {str(e)}")

def render_graph_page():
    st.title("🔗 Citation Graph Explorer")
    
    # Get available papers
    papers = get_papers()
    if not papers:
        st.warning("No papers available. Please upload or fetch papers first.")
        return
    
    # Paper selection
    paper_titles = {p['title']: p['id'] for p in papers}
    selected_titles = st.multiselect(
        "Select papers to visualize in citation graph",
        options=list(paper_titles.keys()),
        max_selections=5  # Limit to prevent overloading
    )
    
    # Depth selection
    depth = st.slider("Citation exploration depth", min_value=1, max_value=3, value=1,
                     help="Higher values will explore deeper citation connections but takes longer")
    
    if selected_titles and st.button("Generate Citation Graph", type="primary"):
        try:
            selected_paper_ids = [paper_titles[title] for title in selected_titles]
            
            with st.spinner("Generating citation graph..."):
                graph_path = build_graph(selected_paper_ids, depth=depth)
                
                # Read and display HTML
                with open(graph_path, 'r') as f:
                    html_content = f.read()
                    display_html(html_content, height=700)
                    
                st.download_button(
                    label="Download Graph HTML",
                    data=html_content,
                    file_name="citation_graph.html",
                    mime="text/html"
                )
                
        except Exception as e:
            st.error(f"Error generating citation graph: {str(e)}")

def render_equation_page():
    st.title("🧮 Equation Simplifier")
    
    st.markdown("""
    ### Simplify and explain LaTeX equations
    Enter a LaTeX equation below and the system will simplify and explain it.
    """)
    
    latex_input = st.text_area(
        "LaTeX Equation (e.g., `\\frac{a^2 + b^2 + 2ab}{(a+b)^2}`)",
        height=100
    )
    
    # Add a reference guide
    with st.expander("LaTeX Reference Guide"):
        st.markdown("""
        ### Common LaTeX Syntax
        - Fractions: `\\frac{numerator}{denominator}`
        - Powers: `x^{exponent}` or `x^n`
        - Subscripts: `x_{subscript}`
        - Square root: `\\sqrt{expression}`
        - Nth root: `\\sqrt[n]{expression}`
        - Greek letters: `\\alpha`, `\\beta`, `\\gamma`, etc.
        - Integrals: `\\int_{lower}^{upper} expression`
        - Summation: `\\sum_{lower}^{upper} expression`
        - Product: `\\prod_{lower}^{upper} expression`
        """)
    
    if latex_input and st.button("Simplify", type="primary"):
        try:
            with st.spinner("Simplifying equation..."):
                # Use the equation mode of the query engine
                result = answer_query(latex_input, mode="equation")
                
                st.markdown("### Simplified Result")
                st.text(result["answer"])
                
                st.markdown("### Mathematical Explanation")
                explanation_query = f"Explain this equation in plain English: {latex_input}"
                explanation = answer_query(explanation_query)
                st.markdown(explanation["answer"])
                
        except Exception as e:
            st.error(f"Error processing equation: {str(e)}")

def render_summary_page():
    st.title("📄 Paper Summarizer")
    
    # Get available papers
    papers = get_papers()
    if not papers:
        st.warning("No papers available. Please upload or fetch papers first.")
        return
    
    # Paper selection
    paper_titles = {p['title']: p['id'] for p in papers}
    
    tab1, tab2 = st.tabs(["Single Paper Summary", "Compare Papers"])
    
    with tab1:
        selected_title = st.selectbox(
            "Select a paper to summarize",
            options=list(paper_titles.keys())
        )
        
        if selected_title:
            paper_id = paper_titles[selected_title]
            
            summary_types = [
                "Technical TL;DR", 
                "Layman's Summary",
                "Key Findings",
                "Methods Summary",
                "Results Analysis"
            ]
            
            summary_type = st.radio("Summary Type", options=summary_types, horizontal=True)
            
            if st.button("Generate Summary", type="primary"):
                try:
                    with st.spinner(f"Generating {summary_type} for {selected_title}..."):
                        summary = generate_summary(paper_id, mode=summary_type.lower().replace("'s", "").replace(" ", "_"))
                        
                        st.markdown(f"### {summary_type}")
                        st.markdown(summary)
                        
                except Exception as e:
                    st.error(f"Error generating summary: {str(e)}")
    
    with tab2:
        selected_titles = st.multiselect(
            "Select papers to compare (2-3 recommended)",
            options=list(paper_titles.keys())
        )
        
        if len(selected_titles) >= 2:
            paper_ids = [paper_titles[title] for title in selected_titles]
            
            comparison_aspects = [
                "Overall Approach", 
                "Methodology Differences",
                "Key Results", 
                "Strengths & Weaknesses"
            ]
            
            aspect = st.radio("Comparison Aspect", options=comparison_aspects, horizontal=True)
            
            if st.button("Generate Comparison", type="primary"):
                try:
                    with st.spinner(f"Comparing papers on {aspect}..."):
                        comparison = compare_papers(paper_ids, aspect=aspect.lower().replace(" & ", "_").replace(" ", "_"))
                        
                        st.markdown(f"### Comparison: {aspect}")
                        st.markdown(comparison)
                        
                except Exception as e:
                    st.error(f"Error generating comparison: {str(e)}")

if __name__ == "__main__":
    main()
