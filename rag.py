import os
import PyPDF2
import nltk
import numpy as np
import logging
from typing import List, Tuple, Dict
from sentence_transformers import SentenceTransformer
from rank_bm25 import BM25Okapi
import faiss
from nltk.tokenize import sent_tokenize, word_tokenize
from transformers import pipeline
import streamlit as st

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Download required NLTK data
logger.info("Downloading NLTK punkt tokenizer data...")
nltk.download('punkt', quiet=True)

class FinancialRAG:
    def __init__(self, model_name: str = "all-MiniLM-L6-v2"):
        """
        Initialize the Financial RAG system with specified models and empty data structures.
        
        Args:
            model_name (str): Name of the sentence transformer model to use for embeddings
        """
        logger.info(f"Initializing FinancialRAG with model: {model_name}")
        
        # Initialize the sentence transformer model for embeddings
        logger.info("Loading sentence transformer model...")
        self.embedding_model = SentenceTransformer(model_name)
        
        # Initialize data structures
        self.chunks = []  # Store text chunks
        self.embeddings = None  # Store embeddings
        self.faiss_index = None  # Vector similarity index
        self.bm25 = None  # Keyword-based search index
        
        # Initialize QA pipeline for answer generation
        logger.info("Loading QA pipeline...")
        self.qa_pipeline = pipeline(
            "question-answering",
            model="deepset/minilm-uncased-squad2",
            device=-1  # CPU
        )
        logger.info("FinancialRAG initialization complete")
        
    def process_pdf(self, pdf_path: str, chunk_size: int = 500):
        """
        Process PDF document and prepare it for question answering.
        
        Args:
            pdf_path (str): Path to the PDF file
            chunk_size (int): Maximum size of text chunks in characters
        """
        logger.info(f"Processing PDF file: {pdf_path}")
        
        # Read PDF file
        try:
            with open(pdf_path, 'rb') as file:
                pdf_reader = PyPDF2.PdfReader(file)
                logger.info(f"PDF loaded successfully. Total pages: {len(pdf_reader.pages)}")
                
                # Extract text from all pages
                text = ""
                for i, page in enumerate(pdf_reader.pages, 1):
                    text += page.extract_text()
                    if i % 10 == 0:  # Log progress every 10 pages
                        logger.info(f"Processed {i} pages...")
        except Exception as e:
            logger.error(f"Error reading PDF file: {e}")
            raise
        
        # Split text into sentences
        logger.info("Splitting text into sentences...")
        sentences = sent_tokenize(text)
        logger.info(f"Total sentences extracted: {len(sentences)}")
        
        # Create chunks of approximately chunk_size characters
        logger.info(f"Creating chunks of maximum {chunk_size} characters...")
        current_chunk = ""
        for sentence in sentences:
            if len(current_chunk) + len(sentence) <= chunk_size:
                current_chunk += " " + sentence
            else:
                if current_chunk:
                    self.chunks.append(current_chunk.strip())
                current_chunk = sentence
        
        if current_chunk:
            self.chunks.append(current_chunk.strip())
        
        logger.info(f"Created {len(self.chunks)} chunks")
        
        # Create BM25 index for keyword search
        logger.info("Creating BM25 index...")
        tokenized_chunks = [word_tokenize(chunk.lower()) for chunk in self.chunks]
        self.bm25 = BM25Okapi(tokenized_chunks)
        
        # Create FAISS index for dense vector search
        logger.info("Creating FAISS index...")
        self.embeddings = self.embedding_model.encode(self.chunks)
        dimension = self.embeddings.shape[1]
        self.faiss_index = faiss.IndexFlatL2(dimension)
        self.faiss_index.add(np.array(self.embeddings).astype('float32'))
        logger.info("Document processing complete")
    
    def hybrid_search(self, query: str, k: int = 3, alpha: float = 0.5) -> List[Tuple[str, float]]:
        """
        Perform hybrid search using both BM25 and dense embeddings.
        
        Args:
            query (str): The search query
            k (int): Number of top results to return
            alpha (float): Weight for combining dense and sparse scores (0.5 = equal weight)
            
        Returns:
            List[Tuple[str, float]]: List of (chunk, score) pairs
        """
        logger.info(f"Performing hybrid search for query: {query}")
        
        # BM25 keyword search
        logger.info("Computing BM25 scores...")
        tokenized_query = word_tokenize(query.lower())
        bm25_scores = self.bm25.get_scores(tokenized_query)
        bm25_scores = (bm25_scores - np.min(bm25_scores)) / (np.max(bm25_scores) - np.min(bm25_scores))
        
        # Dense embedding search
        logger.info("Computing dense embedding scores...")
        query_vector = self.embedding_model.encode([query])[0]
        distances, indices = self.faiss_index.search(
            np.array([query_vector]).astype('float32'), 
            k=len(self.chunks)
        )
        
        # Normalize distances to scores
        dense_scores = 1 / (1 + distances[0])
        dense_scores = (dense_scores - np.min(dense_scores)) / (np.max(dense_scores) - np.min(dense_scores))
        
        # Combine scores
        logger.info("Combining BM25 and dense embedding scores...")
        combined_scores = []
        for idx, (dense_score, bm25_score) in enumerate(zip(dense_scores, bm25_scores)):
            combined_score = alpha * dense_score + (1 - alpha) * bm25_score
            combined_scores.append((self.chunks[idx], combined_score))
        
        # Sort and return top k results
        results = sorted(combined_scores, key=lambda x: x[1], reverse=True)[:k]
        logger.info(f"Retrieved top {k} results")
        return results
    
    def validate_query(self, query: str) -> bool:
        """
        Input-side guardrail to validate queries.
        
        Args:
            query (str): The user's query
            
        Returns:
            bool: Whether the query is valid
        """
        logger.info(f"Validating query: {query}")
        
        # Check query length
        if len(query.strip()) < 10:
            logger.warning("Query rejected: too short")
            return False
        
        # Check for financial keywords
        financial_keywords = {
            'revenue', 'profit', 'loss', 'margin', 'growth', 'sales',
            'earnings', 'cost', 'expense', 'income', 'financial',
            'quarter', 'year', 'annual', 'report', 'balance', 'sheet',
            'cash', 'flow', 'assets', 'liabilities', 'equity'
        }
        
        query_words = set(word.lower() for word in word_tokenize(query))
        if not any(keyword in query_words for keyword in financial_keywords):
            logger.warning("Query rejected: no financial keywords found")
            return False
        
        logger.info("Query validation passed")
        return True
    
    def answer_question(self, query: str) -> Dict:
        """
        Generate answer for a given query.
        
        Args:
            query (str): The user's question
            
        Returns:
            Dict: Contains answer, confidence score, and relevant context
        """
        logger.info(f"Processing question: {query}")
        
        # Validate query
        if not self.validate_query(query):
            logger.warning("Query failed validation")
            return {
                "answer": "I apologize, but I can only answer questions related to Atlassian's financial information.",
                "confidence": 0.0,
                "relevant_context": []
            }
        
        # Get relevant contexts
        logger.info("Retrieving relevant context...")
        relevant_chunks = self.hybrid_search(query, k=3)
        
        # Combine chunks for context
        context = " ".join([chunk[0] for chunk in relevant_chunks])
        
        # Generate answer using QA pipeline
        logger.info("Generating answer using QA pipeline...")
        qa_result = self.qa_pipeline(
            question=query,
            context=context
        )
        
        logger.info(f"Answer generated with confidence score: {qa_result['score']}")
        return {
            "answer": qa_result["answer"],
            "confidence": qa_result["score"],
            "relevant_context": [(chunk[0], chunk[1]) for chunk in relevant_chunks]
        }
    
# Configure Streamlit page settings
st.set_page_config(
    page_title="Atlassian Financial RAG",
    page_icon="💰",
    layout="wide"
)

@st.cache_resource
def initialize_rag():
    """
    Initialize the RAG system with caching to prevent reloading on each rerun.
    
    Returns:
        FinancialRAG: Initialized RAG system or None if initialization fails
    """
    logger.info("Initializing RAG system...")
    try:
        rag = FinancialRAG()
        # Check for PDF file
        pdf_path = "atlassian_annual_report.pdf"
        if os.path.exists(pdf_path):
            logger.info(f"Found PDF file at {pdf_path}")
            rag.process_pdf(pdf_path)
            logger.info("RAG system initialized successfully")
            return rag
        else:
            logger.error(f"PDF file not found at {pdf_path}")
            st.error("PDF file not found. Please ensure the Atlassian annual report is in the atlassian_annual_report.pdf path.")
            return None
    except Exception as e:
        logger.error(f"Error initializing RAG system: {e}")
        st.error(f"Failed to initialize the system: {str(e)}")
        return None

# Main title and description
st.title("Atlassian Financial Information Assistant")
st.markdown("""
This application uses RAG (Retrieval Augmented Generation) to answer questions about Atlassian's financial data.
It combines keyword-based search (BM25) with dense embeddings for accurate information retrieval.
""")

# Initialize RAG system
logger.info("Starting application...")
rag = initialize_rag()

if rag:
    # Query input
    query = st.text_input(
        "Ask a question about Atlassian's financials:",
        placeholder="E.g., What was Atlassian's revenue in the last fiscal year?"
    )

    if query:
        logger.info(f"Processing user query: {query}")
        with st.spinner("Processing your question..."):
            try:
                # Get answer from RAG system
                result = rag.answer_question(query)
                
                # Display answer section
                st.markdown("### Answer")
                st.write(result["answer"])
                
                # Display confidence score with progress bar
                confidence = result["confidence"]
                st.progress(confidence)
                st.caption(f"Confidence Score: {confidence:.2%}")
                
                # Display supporting evidence
                st.markdown("### Supporting Evidence")
                for i, (context, score) in enumerate(result["relevant_context"], 1):
                    with st.expander(f"Evidence #{i} - Relevance Score: {score:.2%}"):
                        st.write(context)
                
                # Show warning for low confidence answers
                if confidence < 0.5:
                    logger.warning(f"Low confidence answer generated: {confidence:.2%}")
                    st.warning("⚠️ This answer has low confidence. Please verify the information from other sources.")
                
                logger.info("Query processed successfully")
                
            except Exception as e:
                logger.error(f"Error processing query: {e}")
                st.error(f"An error occurred while processing your question: {str(e)}")

    # Display usage tips
    st.markdown("---")
    st.markdown("""
    ### Tips for better results:
    - Ask specific questions about financial metrics
    - Include time periods in your questions
    - Focus on information that would be found in an annual report
    
    ### Example questions:
    1. What was Atlassian's total revenue in fiscal year 2023?
    2. How much did operating expenses increase compared to the previous year?
    3. What is the company's cash position at the end of the fiscal year?
    """)
else:
    logger.error("Application failed to initialize - RAG system not available")
    st.error("System not initialized. Please check if the PDF file is available.") 