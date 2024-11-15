from typing import Dict, List, Any
import os
from openai import OpenAI
from pinecone import Pinecone
from typing import Optional
import logging
from langchain_core.tools import tool
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class RAGTools:
    def __init__(self):
        self.openai_client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))
        self.pc = Pinecone(api_key=os.getenv("PINECONE_API_KEY"))
        self.index = self.pc.Index("researchagent")
        self.namespace = "investment_research"
        
    def create_embedding(self, text: str) -> List[float]:
        """Create embedding for the input text"""
        try:
            embedding = self.openai_client.embeddings.create(
                input=text,
                model="text-embedding-3-small"
            ).data[0].embedding
            return embedding
        except Exception as e:
            logger.error(f"Error creating embedding: {e}")
            raise

    def process_matches(self, matches: List[Dict]) -> List[Dict[str, str]]:
        """Process and format the matches from Pinecone"""
        results = []
        for match in matches:
            if not hasattr(match, 'metadata') or not match.metadata:
                continue
                
            metadata = match.metadata
            result = {
                "content": metadata.get('text', ''),
                "source": metadata.get('doc_name', ''),
                "content_type": metadata.get('content_type', ''),
                "score": match.score
            }
            results.append(result)
        return results

@tool("RAGSearch")
async def rag_search(query: str) -> List[Dict[str, str]]:
    """Performs RAG search using embeddings and Pinecone
    Args:
        query (str): Search query
    Returns:
        List[Dict]: List of relevant documents with content and metadata
    """
    try:
        rag = RAGTools()
        
        # Create query embedding
        query_embedding = rag.create_embedding(query)
        
        # Query Pinecone
        result = rag.index.query(
            vector=query_embedding,
            top_k=5,
            include_metadata=True,
            namespace=rag.namespace
        )
        
        # Process results
        processed_results = rag.process_matches(result.matches)
        logger.info(f"Retrieved {len(processed_results)} results for query: {query}")
        
        return processed_results

    except Exception as e:
        logger.error(f"RAG search error: {str(e)}")
        return []

@tool("RAGFilteredSearch")
async def rag_filtered_search(query: str, content_type: str) -> List[Dict[str, str]]:
    """Performs filtered RAG search using embeddings and Pinecone
    Args:
        query (str): Search query
        content_type (str): Type of content to filter (e.g., 'text', 'table_text', 'picture')
    Returns:
        List[Dict]: List of relevant documents with content and metadata
    """
    try:
        rag = RAGTools()
        
        # Create query embedding
        query_embedding = rag.create_embedding(query)
        
        # Create filter
        filter_expression = {
            "content_type": {"$eq": content_type}
        }
        
        # Query Pinecone with filter
        result = rag.index.query(
            vector=query_embedding,
            filter=filter_expression,
            top_k=5,
            include_metadata=True,
            namespace=rag.namespace
        )
        
        # Process results
        processed_results = rag.process_matches(result.matches)
        logger.info(f"Retrieved {len(processed_results)} filtered results for query: {query} and type: {content_type}")
        
        return processed_results

    except Exception as e:
        logger.error(f"RAG filtered search error: {str(e)}")
        return []