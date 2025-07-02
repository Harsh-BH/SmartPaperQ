import networkx as nx
from pyvis.network import Network
import requests
import json
from typing import List, Dict, Any
import os
from utils.config import VECTOR_STORE_PATH

class CitationGraph:
    def __init__(self, depth: int = 1):
        """
        Initialize the citation graph builder
        
        Args:
            depth: How many levels of citations to include
        """
        self.graph = nx.DiGraph()
        self.depth = depth
        self.paper_metadata = {}
    
    def _fetch_citations(self, paper_id: str) -> List[str]:
        """
        Fetch citations for a paper from Semantic Scholar API
        """
        try:
            url = f"https://api.semanticscholar.org/v1/paper/arXiv:{paper_id}"
            response = requests.get(url)
            data = response.json()
            
            # Store paper metadata
            self.paper_metadata[paper_id] = {
                'title': data.get('title', 'Unknown'),
                'authors': [a.get('name') for a in data.get('authors', [])],
                'year': data.get('year'),
                'venue': data.get('venue')
            }
            
            # Get citation IDs
            citations = []
            for cited in data.get('citations', []):
                if 'arxivId' in cited:
                    citations.append(cited['arxivId'])
            
            return citations
        except Exception as e:
            print(f"Error fetching citations for {paper_id}: {e}")
            return []
    
    def build_graph(self, paper_ids: List[str]):
        """
        Build and visualize citation graph for a list of papers
        
        Args:
            paper_ids: List of arXiv paper IDs to analyze
        """
        # Add initial papers as nodes
        for paper_id in paper_ids:
            self.graph.add_node(paper_id, color='red', size=25, title=paper_id)
            
        # Process citations up to the specified depth
        papers_to_process = paper_ids.copy()
        processed = set()
        
        for _ in range(self.depth):
            next_level = []
            for paper_id in papers_to_process:
                if paper_id in processed:
                    continue
                    
                citations = self._fetch_citations(paper_id)
                for cited_id in citations:
                    # Add cited paper as node if not exists
                    if not self.graph.has_node(cited_id):
                        self.graph.add_node(cited_id, color='blue', size=15, 
                                           title=self.paper_metadata.get(cited_id, {}).get('title', cited_id))
                    
                    # Add citation edge
                    self.graph.add_edge(paper_id, cited_id, arrows='to')
                    next_level.append(cited_id)
                
                processed.add(paper_id)
            
            papers_to_process = next_level
    
    def visualize(self, output_path: str = "citation_graph.html"):
        """
        Generate interactive visualization of the citation graph
        
        Args:
            output_path: Path to save the HTML visualization
        """
        net = Network(height="750px", width="100%", directed=True, notebook=False)
        
        # Copy graph from networkx
        net.from_nx(self.graph)
        
        # Set physics layout
        net.toggle_physics(True)
        net.show_buttons(filter_=['physics'])
        
        # Save visualization
        net.save_graph(output_path)
        return output_path
    
    def get_most_cited(self, limit: int = 5) -> List[Dict[str, Any]]:
        """
        Get the most cited papers in the graph
        
        Returns:
            List of {paper_id, count, title} dictionaries
        """
        in_degree = dict(self.graph.in_degree())
        top_cited = sorted(in_degree.items(), key=lambda x: x[1], reverse=True)[:limit]
        
        results = []
        for paper_id, count in top_cited:
            metadata = self.paper_metadata.get(paper_id, {})
            results.append({
                'paper_id': paper_id,
                'citation_count': count,
                'title': metadata.get('title', 'Unknown'),
                'authors': metadata.get('authors', []),
                'year': metadata.get('year')
            })
        
        return results

def build_graph(paper_ids: List[str], depth: int = 1) -> str:
    """
    Build and visualize citation graph for specified papers
    
    Args:
        paper_ids: List of paper IDs to include
        depth: Citation depth to explore
    
    Returns:
        Path to the generated HTML visualization
    """
    graph = CitationGraph(depth=depth)
    graph.build_graph(paper_ids)
    return graph.visualize()
