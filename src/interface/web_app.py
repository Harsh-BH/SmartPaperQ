import streamlit as st
import os
import tempfile
import time
from dotenv import load_dotenv
import threading
import concurrent.futures

# Replace the signal-based timeout with a threading-based solution
class TimeoutError(Exception):
    pass

def run_with_timeout(func, args=(), kwargs={}, timeout=30):
    """Run a function with a timeout using threading instead of signals"""
    result = {"completed": False, "result": None, "error": None}
    
    def worker():
        try:
            result["result"] = func(*args, **kwargs)
            result["completed"] = True
        except Exception as e:
            result["error"] = str(e)
    
    thread = threading.Thread(target=worker)
    thread.daemon = True  # Make thread a daemon so it dies when main thread dies
    
    thread.start()
    thread.join(timeout)
    
    if thread.is_alive():
        # Thread is still running after timeout
        return {"status": "error", "message": f"Operation timed out after {timeout} seconds"}
    
    if result["completed"]:
        return result["result"]
    elif result["error"]:
        return {"status": "error", "message": result["error"]}
    else:
        return {"status": "error", "message": "Unknown error occurred"}

# Load environment variables
load_dotenv()

try:
    from src.models.rag_pipeline import RAGPipeline
    from src.services.arxiv_api import ArxivService
    from src.services.paper_comparison import PaperComparison
    services_loaded = True
except ImportError as e:
    services_loaded = False
    import_error = str(e)

from src.utils.direct_download import direct_download_paper

def create_app():
    st.set_page_config(page_title="Paper Shaper", page_icon="📚", layout="wide")
    
    if not services_loaded:
        st.error(f"Error loading services: {import_error}")
        st.info("Please make sure all required packages are installed by running `python setup.py`")
        return
    
    # Initialize session state variables only if not present
    for key, default in [
        ("rag_pipeline", RAGPipeline()),
        ("processed", False),
        ("history", []),
        ("downloaded_papers", []),
        ("download_expander_open", False),
        ("download_bytes", {}),
        ("download_in_progress", {}),
        ("download_completed", {}),
        ("search_results", []),
    ]:
        if key not in st.session_state:
            st.session_state[key] = default

    arxiv_service = ArxivService()
    paper_comparison = PaperComparison()
    download_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), "../../downloads"))
    os.makedirs(download_dir, exist_ok=True)

    # Header
    col1, col2 = st.columns([3, 1])
    with col1:
        st.title("📚 Paper Shaper")
        st.markdown("##### AI-Powered Research Paper Assistant")
    with col2:
        st.markdown("")
        st.markdown("")
        if st.session_state.processed:
            st.success("Document Ready")

    # Sidebar for file upload and settings
    with st.sidebar:
        st.header("Document Manager")
        uploaded_file = st.file_uploader("Upload Research Paper", type="pdf", key="sidebar_uploader")
        use_cache = st.checkbox("Use cache (faster for repeated uploads)", value=True)
        timeout_seconds = st.slider("Processing timeout (seconds)", 10, 300, 60)
        
        if "processing_status" not in st.session_state:
            st.session_state.processing_status = ""
        
        if uploaded_file is not None:
            with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as tmp:
                tmp.write(uploaded_file.getvalue())
                tmp_path = tmp.name
            
            if st.button("Process Paper", key="sidebar_process"):
                try:
                    st.session_state.processing_status = "Starting processing..."
                    progress_bar = st.progress(0)
                    status_text = st.empty()
                    
                    # Processing steps
                    status_text.text("Step 1/4: Reading PDF content")
                    progress_bar.progress(25)
                    
                    # Use the thread-safe timeout function instead of the signal-based approach
                    status_text.text("Step 2/4: Extracting text chunks")
                    progress_bar.progress(50)
                    
                    status_text.text("Step 3/4: Creating embeddings")
                    progress_bar.progress(75)
                    
                    # Run document processing with a timeout
                    result = run_with_timeout(
                        st.session_state.rag_pipeline.process_document, 
                        args=(tmp_path,), 
                        kwargs={"use_cache": use_cache},
                        timeout=timeout_seconds
                    )
                    
                    status_text.text("Step 4/4: Finalizing")
                    progress_bar.progress(100)
                    
                    # Process result
                    st.session_state.processed = (result.get("status") == "success")
                    
                    if st.session_state.processed:
                        st.success(f"✅ Document processed in {result.get('processing_time', 0):.2f}s")
                        if "chunk_count" in result:
                            st.info(f"📊 Extracted {result['chunk_count']} text chunks")
                    else:
                        st.error(f"❌ Error: {result.get('message', 'Unknown error')}")
                except Exception as e:
                    st.error(f"Error: {str(e)}")
                    st.session_state.processed = False
                finally:
                    # Always clean up the temporary file
                    try:
                        os.unlink(tmp_path)
                    except:
                        pass
                    
            # Update the Simple Process button to use the thread-safe timeout
            if st.button("Simple Process (for large PDFs)"):
                try:
                    st.info("Using simplified processing with larger chunks...")
                    
                    # Set parameters for larger chunks
                    chunk_size = 2000
                    chunk_overlap = 100
                    
                    # Display progress indicators
                    progress_bar = st.progress(0)
                    status_text = st.empty()
                    status_text.text("Processing document with simplified settings...")
                    progress_bar.progress(25)
                    
                    # Run with extended timeout (60 seconds)
                    result = run_with_timeout(
                        st.session_state.rag_pipeline.process_document, 
                        args=(tmp_path,), 
                        kwargs={"use_cache": True, "chunk_size": chunk_size, "chunk_overlap": chunk_overlap},
                        timeout=60  # Use a fixed timeout for the simple process
                    )
                    
                    progress_bar.progress(100)
                    
                    st.session_state.processed = (result.get("status") == "success")
                    if st.session_state.processed:
                        st.success(f"Document processed in simplified mode ({result.get('chunk_count', 0)} chunks)")
                    else:
                        st.error(f"Error: {result.get('message', 'Unknown error')}")
                except Exception as e:
                    st.error(f"Processing error: {str(e)}")
                finally:
                    try:
                        os.unlink(tmp_path)
                    except:
                        pass

        st.markdown("---")
        st.markdown("### Model Settings")
        model_options = {
            "llama3-70b-8192": "Llama 3 70B (fastest)",
            "llama3-8b-8192": "Llama 3 8B (balanced)",
            "mixtral-8x7b-32768": "Mixtral 8x7B (longest context)"
        }
        selected_model = st.selectbox(
            "Select LLM Model",
            options=list(model_options.keys()),
            format_func=lambda x: model_options[x],
            index=0
        )
        if selected_model != getattr(st.session_state.rag_pipeline.generator, "model", "llama3-70b-8192"):
            st.session_state.rag_pipeline.generator.model = selected_model
            st.success(f"Model switched to {selected_model}")

    # Tabs
    tabs = st.tabs(["Search arXiv", "Paper Comparison", "Upload Papers", "Ask Questions"])

    # Tab 1: Search arXiv
    with tabs[0]:
        st.header("Search arXiv Papers")
        with st.form("search_form"):
            search_query = st.text_input("Enter search query")
            max_results = st.slider("Maximum number of results", 1, 50, 10)
            submit_search = st.form_submit_button("Search")
        if submit_search and search_query:
            with st.spinner("Searching arXiv..."):
                results = arxiv_service.search_papers(search_query, max_results)
            if results:
                st.success(f"Found {len(results)} papers matching your query")
                st.session_state.search_results = results
            else:
                st.warning("No papers found matching your query")

        # Show search results
        for i, paper in enumerate(st.session_state.search_results):
            with st.expander(f"{i+1}. {paper['title']}"):
                st.write(f"**Authors:** {', '.join(paper['authors'])}")
                st.write(f"**Published:** {paper['published'].strftime('%Y-%m-%d')}")
                st.write(f"**Summary:** {paper['summary']}")
                paper_id = paper['entry_id']
                already_in_list = any(p.get('entry_id') == paper_id for p in st.session_state.downloaded_papers)
                col1, col2 = st.columns([1, 3])

                # Download PDF Button (callback)
                def download_pdf_callback(paper_id=paper_id, idx=i):
                    download_result = direct_download_paper(paper_id, download_dir)
                    if download_result["success"]:
                        with open(download_result["output_path"], "rb") as f:
                            st.session_state.download_bytes[f"paper_{paper_id}_{idx}"] = f.read()
                        st.session_state.download_completed[f"paper_{paper_id}_{idx}"] = True
                        st.session_state.download_in_progress[f"paper_{paper_id}_{idx}"] = False
                    else:
                        st.session_state.download_completed[f"paper_{paper_id}_{idx}"] = False
                        st.session_state.download_in_progress[f"paper_{paper_id}_{idx}"] = False
                        st.error(f"❌ Download failed: {download_result.get('error', 'Unknown error')}")

                with col1:
                    st.write("Download Options")
                    paper_key = f"paper_{paper_id}_{i}"
                    if paper_key in st.session_state.download_bytes:
                        st.download_button(
                            label=f"Save {paper_id}.pdf",
                            data=st.session_state.download_bytes[paper_key],
                            file_name=f"{paper_id}.pdf",
                            mime="application/pdf",
                            key=f"save_direct_{i}"
                        )
                        st.success("✅ Download successful!")
                    else:
                        st.button(
                            "Download PDF",
                            key=f"direct_dl_{i}",
                            on_click=download_pdf_callback,
                            args=(paper_id, i)
                        )

                # Add to comparison Button (callback)
                def add_to_comparison_callback(paper=paper, paper_id=paper_id):
                    # Download the paper if not already downloaded
                    download_result = direct_download_paper(paper_id, download_dir)
                    if download_result["success"]:
                        comparison_paper = {
                            'entry_id': paper_id,
                            'title': paper['title'],
                            'authors': paper['authors'],
                            'published': paper['published'],
                            'summary': paper['summary'],
                            'local_path': download_result["output_path"]
                        }
                        if not any(p.get('entry_id') == paper_id for p in st.session_state.downloaded_papers):
                            st.session_state.downloaded_papers.append(comparison_paper)
                            st.success(f"Added '{paper['title']}' to comparison list!")
                        else:
                            st.info("✅ Already in comparison list")
                    else:
                        st.error(f"Failed to download: {download_result.get('error', 'Unknown error')[:100]}...")

                with col2:
                    if already_in_list:
                        st.info("✅ Already in comparison list")
                    else:
                        st.button(
                            "Add to comparison",
                            key=f"add_comp_{i}",
                            on_click=add_to_comparison_callback
                        )
                st.markdown(f"[View on arXiv](https://arxiv.org/abs/{paper_id})")

    # Tab 2: Paper Comparison
    with tabs[1]:
        st.header("Compare Papers")
        st.info(f"You have {len(st.session_state.downloaded_papers)} papers in your comparison list.")
        if len(st.session_state.downloaded_papers) < 2:
            st.warning("Please add at least two papers to the comparison list from the Search tab")
        else:
            st.subheader("Available Papers")
            for i, paper in enumerate(st.session_state.downloaded_papers):
                paper_title = paper.get('title', f"Paper {i+1}")
                paper_path = paper.get('local_path', 'No path')
                with st.expander(f"{i+1}. {paper_title}"):
                    st.write(f"**Title:** {paper_title}")
                    st.write(f"**Authors:** {', '.join(paper.get('authors', ['Unknown']))}")
                    st.write(f"**File:** {os.path.basename(paper_path)}")
                    if st.button(f"Remove from list", key=f"remove_{i}"):
                        st.session_state.downloaded_papers.pop(i)
                        st.success(f"Removed '{paper_title}' from comparison list")
                        st.experimental_rerun()  # Needed to update UI after removal

            # Paper selection for comparison
            col1, col2 = st.columns(2)
            with col1:
                paper1_idx = st.selectbox(
                    "Select first paper", 
                    range(len(st.session_state.downloaded_papers)),
                    format_func=lambda i: st.session_state.downloaded_papers[i]['title']
                )
            with col2:
                paper2_idx = st.selectbox(
                    "Select second paper", 
                    range(len(st.session_state.downloaded_papers)),
                    format_func=lambda i: st.session_state.downloaded_papers[i]['title'],
                    index=min(1, len(st.session_state.downloaded_papers)-1)
                )
            if st.button("Compare Papers", key="compare_btn"):
                paper1 = st.session_state.downloaded_papers[paper1_idx]
                paper2 = st.session_state.downloaded_papers[paper2_idx]
                if paper1_idx == paper2_idx:
                    st.error("Please select two different papers for comparison")
                else:
                    with st.spinner("Analyzing and comparing papers..."):
                        comparison_results = paper_comparison.compare_papers(
                            paper1['local_path'],
                            paper2['local_path']
                        )
                    st.subheader("Comparison Results")
                    similarity = comparison_results['similarity_score'] * 100
                    st.metric("Similarity Score", f"{similarity:.1f}%")
                    st.subheader("Common Keywords")
                    if comparison_results['common_keywords']:
                        st.write(", ".join(comparison_results['common_keywords']))
                    else:
                        st.write("No significant common keywords found")
                    col1, col2 = st.columns(2)
                    with col1:
                        st.subheader(f"Paper 1: {paper1['title']}")
                        st.write(f"Authors: {', '.join(paper1['authors'])}")
                        st.write(f"Length: {comparison_results['paper1_length']} characters")
                    with col2:
                        st.subheader(f"Paper 2: {paper2['title']}")
                        st.write(f"Authors: {', '.join(paper2['authors'])}")
                        st.write(f"Length: {comparison_results['paper2_length']} characters")

    # Tab 3: Upload Papers
    with tabs[2]:
        st.header("Upload Your Papers")
        uploaded_file = st.file_uploader("Choose a PDF file", type="pdf", key="tab3_uploader")
        if uploaded_file is not None:
            st.success("File uploaded successfully!")
            file_path = os.path.join(download_dir, uploaded_file.name)
            with open(file_path, "wb") as f:
                f.write(uploaded_file.getbuffer())
            already_exists = any(p.get('title') == uploaded_file.name for p in st.session_state.downloaded_papers)
            if not already_exists:
                paper_info = {
                    'title': uploaded_file.name,
                    'authors': ["User Upload"],
                    'local_path': file_path,
                    'entry_id': f"upload_{len(st.session_state.downloaded_papers)}",
                }
                st.session_state.downloaded_papers.append(paper_info)
                st.success("Added to comparison list!")
            else:
                st.info("This paper is already in your comparison list.")

    # Tab 4: Ask Questions
    with tabs[3]:
        st.header("Ask Questions About Papers")
        if len(st.session_state.downloaded_papers) > 0:
            paper_options = [p['title'] for p in st.session_state.downloaded_papers]
            selected_paper_idx = st.selectbox("Select paper to query:", range(len(paper_options)), 
                                            format_func=lambda i: paper_options[i])
            selected_paper = st.session_state.downloaded_papers[selected_paper_idx]
            st.write(f"Selected paper: **{selected_paper['title']}**")
            process_query_paper = st.button("Process Paper for Questions")
            if process_query_paper:
                with st.spinner("Processing paper for question answering..."):
                    result = st.session_state.rag_pipeline.process_document(
                        selected_paper['local_path'], 
                        use_cache=True
                    )
                    st.session_state.processed = (result["status"] == "success")
                    if st.session_state.processed:
                        st.success(f"Paper processed successfully in {result['processing_time']:.2f}s")
                    else:
                        st.error(f"Error processing paper: {result['message']}")
            if st.session_state.processed:
                query = st.text_input("Ask a question about the paper:", key="query_input")
                col1, col2 = st.columns([1, 5])
                with col1:
                    submit_button = st.button("Ask", type="primary")
                with col2:
                    clear_button = st.button("Clear History")
                if submit_button and query:
                    with st.spinner("Generating answer..."):
                        start_time = time.time()
                        result = st.session_state.rag_pipeline.answer_question(query)
                        total_time = time.time() - start_time
                        if result["status"] == "success":
                            st.session_state.history.append({
                                "query": query,
                                "answer": result["answer"],
                                "sources": result["sources"],
                                "time": total_time
                            })
                            st.success("Answer generated!")
                        else:
                            st.error(result["message"])
                if clear_button:
                    st.session_state.history = []
                    st.success("History cleared!")
                if st.session_state.history:
                    for i, item in enumerate(reversed(st.session_state.history)):
                        with st.container():
                            st.markdown(f"### Question {len(st.session_state.history) - i}")
                            st.info(item["query"])
                            st.markdown("### Answer")
                            st.markdown(item["answer"])
                            with st.expander(f"View Sources (Response time: {item['time']:.2f}s)"):
                                for j, (chunk, score) in enumerate(item["sources"][:3], 1):
                                    st.markdown(f"**Source {j}** (Relevance: {score:.2f})")
                                    st.text(chunk[:300] + "..." if len(chunk) > 300 else chunk)
                            st.markdown("---")
        else:
            st.info("Please add papers through the Search or Upload tabs first.")
            with st.expander("Example questions you can ask after uploading a paper"):
                st.markdown("""
                - What are the main findings of this research?
                - Summarize the methodology used in this paper.
                - What are the limitations mentioned in the paper?
                - Explain the significance of figure X in the paper.
                - How does this research compare to prior work in the field?
                - What future work is suggested by the authors?
                """)

    st.markdown("---")
    st.markdown("*Paper Shaper - Analyze, Compare, and Query Research Papers*")
    return st

if __name__ == "__main__":
    app = create_app()
