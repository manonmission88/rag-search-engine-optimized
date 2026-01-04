# RAG Search Engine Optimized

A high-performance movie search engine that combines multiple retrieval and ranking techniques to deliver accurate and relevant search results. This project implements a complete Retrieval-Augmented Generation (RAG) pipeline with advanced search capabilities including keyword search, semantic search, hybrid search, multimodal search, and intelligent query enhancement.

## Overview

This search engine provides a sophisticated approach to information retrieval by combining traditional keyword-based methods with modern neural search techniques. It demonstrates practical implementations of various search algorithms and ranking methods commonly used in production search systems, including state-of-the-art multimodal search that allows users to find movies using images.

## Key Features

### Search Methods

**Keyword Search with BM25**
The system implements BM25 (Best Matching 25), a probabilistic ranking function used extensively in information retrieval. It builds an inverted index from movie documents and ranks results based on term frequency and document frequency statistics.

**Semantic Search**
Utilizes sentence transformers to create vector embeddings of documents and queries. The search uses cosine similarity to find semantically related content, even when exact keywords don't match. Documents are chunked to improve granularity and relevance.

**Hybrid Search**
Combines BM25 and semantic search using two distinct fusion methods:

- **Weighted Search**: Normalizes scores from both methods and combines them using a configurable alpha parameter to balance keyword and semantic relevance.
- **Reciprocal Rank Fusion (RRF)**: A rank-based fusion technique that combines results from multiple retrieval systems without requiring score normalization. Uses the formula: RRF score = 1 / (k + rank), where k is typically set to 60.

**Multimodal Search**
Leverages vision-language models to enable image-based movie search. Upload an image (e.g., a movie poster, scene, or related image) and the system finds semantically similar movies by comparing image embeddings with text embeddings of movie titles and descriptions. Uses CLIP (Contrastive Language-Image Pre-training) models to create a shared embedding space between images and text.

### Query Enhancement

**Spell Correction**
Automatically detects and corrects spelling errors in search queries using the Groq AI API with the Llama 3.3 70B model, ensuring users find relevant results even with typos.

**Query Rewriting**
Transforms vague or colloquial queries into more specific, searchable terms. For example, "that bear movie where leo gets attacked" becomes "The Revenant Leonardo DiCaprio bear attack."

**Query Expansion**
Adds relevant synonyms and related terms to broaden search coverage while maintaining relevance. Helps capture documents that use different terminology.

**Image Query Enhancement**
Analyzes uploaded images along with text queries to generate enhanced search queries. Uses vision-language models to extract visual information from images and combine it with textual context to create more detailed and accurate search queries.

### Result Reranking

**Individual Scoring**
Uses a language model to score each document individually against the query on a 0-10 scale, considering direct relevance and user intent.

**Batch Ranking**
Presents all candidate documents to the language model simultaneously for holistic ranking, allowing for comparative evaluation.

**Cross-Encoder Reranking**
Employs a specialized cross-encoder model (ms-marco-TinyBERT-L2-v2) trained specifically for passage ranking. This method computes relevance scores for query-document pairs and reorders results based on deep semantic understanding.

## Technical Architecture

### Core Components

**Inverted Index**
A data structure mapping terms to the documents containing them, enabling efficient keyword lookup. Stores term frequencies, document frequencies, and supports BM25 scoring.

**Vector Store**
Maintains dense vector representations of document chunks using sentence transformers. Enables similarity-based retrieval through cosine distance computation.

**Multimodal Embedding System**
Creates unified embeddings for both images and text using CLIP models, enabling cross-modal search where users can query with images to find relevant text-based content, or vice versa.

**Hybrid Search Engine**
Orchestrates multiple search strategies, normalizes scores across different scales, and implements fusion algorithms to combine diverse ranking signals.

**Query Processor**
Handles query enhancement through integration with large language models, manages API communication, and includes error handling with fallback mechanisms.

### Key Algorithms

**BM25 Scoring**
Ranking function that considers term frequency (TF), inverse document frequency (IDF), and document length normalization. Parameters b and k1 control the impact of document length and term saturation.

**Cosine Similarity**
Measures the cosine of the angle between two vectors in multidimensional space. Used to compute semantic similarity between query and document embeddings.

**Score Normalization**
Min-max normalization technique that scales scores from different retrieval methods to a common [0, 1] range, enabling fair combination.

**Reciprocal Rank Fusion**
A rank aggregation method that combines multiple rankings by summing reciprocal ranks. More robust than score-based fusion as it doesn't assume score calibration.

## Installation

### Prerequisites

Python 3.13 or higher is required.

### Setup

Clone the repository and navigate to the project directory:

```bash
git clone <repository-url>
cd rag-search-engine-optimized
```

Install dependencies using uv or pip:

```bash
uv sync
```

or

```bash
pip install -e .
```

### Environment Configuration

Create a `.env` file in the project root with your API keys:

```
GROQ_API_KEY=your_groq_api_key_here
GEMINI_API_KEY=your_gemini_api_key_here

```

LLM MODEL is required for query enhancement
(I used groq and gemini API -- free tiers)

## Usage

The system provides three command-line interfaces for different search methods.

### Keyword Search

Perform BM25-based keyword search:

```bash
python cli/keyword_search_cli.py "action movie"
```

Advanced options:

```bash
python cli/keyword_search_cli.py "thriller mystery" --limit 10 --k1 1.5 --b 0.75
```

### Semantic Search

Search using vector embeddings and semantic similarity:

```bash
python cli/semantic_search_cli.py "movies about friendship"
```

Options:

```bash
python cli/semantic_search_cli.py "space exploration" --limit 10
```

### Hybrid Search

#### Weighted Hybrid Search

Combine BM25 and semantic search with adjustable weights:

```bash
python cli/hybrid_search_cli.py weighted-search "adventure film" --alpha 0.5 --limit 10
```

The `--alpha` parameter controls the balance:
- `0.0`: Pure semantic search
- `0.5`: Equal weighting (default)
- `1.0`: Pure BM25 search

#### RRF Hybrid Search

Use Reciprocal Rank Fusion for result combination:

```bash
python cli/hybrid_search_cli.py rrf-search "family movie about bears"
```

With query enhancement:

```bash
python cli/hybrid_search_cli.py rrf-search "briish bear movee" --enhance spell --limit 10
```

With query rewriting:

```bash
python cli/hybrid_search_cli.py rrf-search "that movie with the talking bear" --enhance rewrite
```

With cross-encoder reranking:

```bash
python cli/hybrid_search_cli.py rrf-search "family movie about bears in the woods" --rerank-method cross_encoder --limit 10
```

Parameters:
- `--k`: RRF constant parameter (default: 60)
- `--enhance`: Query enhancement method (`spell`, `rewrite`, `expand`)
- `--rerank-method`: Reranking approach (`individual`, `batch`, `cross_encoder`)
- `--limit`: Number of final results (default: 10)

### Multimodal Search

#### Image-Based Search

Search for movies using images (posters, scenes, or related imagery):

```bash
python cli/multimodal_search_cli.py image_search data/image.png
```

This command analyzes the visual content of the image and finds movies with similar themes, characters, or visual styles by comparing the image embedding against text descriptions of movies in the database.

#### Verify Image Embeddings

Test that image embeddings are working correctly:

```bash
python cli/multimodal_search_cli.py verify_image_embedding data/image.png
```

### Image Query Enhancement

Generate enhanced search queries by analyzing both images and text:

```bash
python cli/describe_image_cli.py --image data/image.png --query "movie about a bear"
```

The system uses vision-language models to analyze the image and combine visual information with the text query to create a more detailed and specific search query.

## How It Works

### Indexing Process

1. **Document Loading**: Movies are loaded from the JSON dataset
2. **Text Processing**: Documents are tokenized and stop words are removed
3. **Index Building**: An inverted index is constructed mapping terms to documents
4. **Vector Encoding**: Documents are chunked and encoded into dense vectors using sentence transformers
5. **Caching**: Embeddings and metadata are cached to disk for fast retrieval

### Search Process

1. **Query Processing**: The user query is optionally enhanced through spell correction, rewriting, or expansion
2. **Retrieval**: Multiple retrieval methods fetch candidate documents
   - BM25 ranks documents by keyword relevance
   - Semantic search finds similar vectors
3. **Fusion**: Results are combined using weighted averaging or reciprocal rank fusion
4. **Reranking**: Optionally, a cross-encoder or LLM reranks the top results
5. **Presentation**: Final ranked results are displayed with scores and metadata

### Caching Strategy

The system automatically caches embeddings to avoid recomputing expensive vector representations. On first run, embeddings are generated and saved. Subsequent runs load from cache, significantly improving performance.

## Key Technologies

**Sentence Transformers**
Open-source framework for state-of-the-art sentence, text, and image embeddings. Used for generating semantic vectors.

**CLIP (Contrastive Language-Image Pre-training)**
A neural network trained on image-text pairs that learns visual concepts from natural language supervision. Enables multimodal search by creating a shared embedding space for images and text. The model used is `clip-ViT-B-32`.

**NLTK**
Natural Language Toolkit for text processing, tokenization, and stop word filtering.

**Cross-Encoder Models**
Transformer models that jointly encode query and document pairs for highly accurate relevance scoring. More computationally expensive but more accurate than bi-encoder approaches.

**Groq API**
Provides access to high-performance language models with extremely fast inference for query enhancement tasks. Uses models like Llama for text processing and vision-language models for image analysis.

**Gemini API**
Google's multimodal AI for advanced image analysis and query generation, enabling sophisticated image-to-text query enhancement.

**NumPy**
Fundamental package for numerical computing, used for efficient vector operations and similarity calculations.

**PIL (Python Imaging Library)**
Used for loading and processing images before embedding generation.

## Models Used

### Text Embeddings
- **all-MiniLM-L6-v2**: Lightweight sentence transformer for semantic search (384 dimensions)
- **ms-marco-TinyBERT-L2-v2**: Cross-encoder model for passage reranking

### Multimodal Embeddings
- **clip-ViT-B-32**: Vision Transformer model for creating unified image-text embeddings (512 dimensions)

### Large Language Models
- **Llama 3.3 70B** (via Groq): Query enhancement, spell correction, and rewriting
- **Llama 4 Scout 17B** (via Groq): Image understanding and multimodal query generation
- **Gemini 2.0 Flash**: Advanced vision-language model for image analysis

## Current Implementation

Currently, this program is designed to work with a **movie dataset**. The system loads movie documents with title and description fields to build indexes and embeddings for comprehensive movie search and discovery.

Similar approaches can be applied for other domains and datasets, including:

- **Product Catalogs**: E-commerce platforms with product names, descriptions, images, and specifications
- **Academic Papers**: Research databases with titles, abstracts, and full-text papers
- **News Articles**: News aggregators with headlines, content, and embedded images
- **Document Repositories**: Enterprise knowledge bases with technical documentation and manuals
- **Image Galleries**: Photo collections with captions, metadata, and visual content
- **Restaurant/Hotel Reviews**: Hospitality platforms with descriptions, amenities, and photos
- **Real Estate Listings**: Property databases with descriptions, images, and location data
- **Video Transcripts**: Video platforms with metadata and searchable transcripts
- **Job Listings**: Career platforms with job descriptions, requirements, and company information
- **Medical Records**: Healthcare systems with patient histories, diagnoses, and clinical notes

The modular architecture allows easy adaptation to any dataset by simply replacing the data loading component and adjusting field mappings.

## Performance Considerations

**Embedding Cache**: Pre-computed embeddings are stored on disk to avoid regeneration
**Chunking Strategy**: Documents are split into semantic chunks to improve retrieval granularity
**Efficient Scoring**: BM25 uses optimized data structures for fast term lookup
**Batch Processing**: Cross-encoder predictions are batched for improved throughput
**Result Limiting**: Search methods fetch more results initially before reranking to ensure quality

## Learning Resources

https://boot.dev

https://nlp.stanford.edu/IR-book/

https://huggingface.co/course

https://www.sbert.net/

https://paperswithcode.com/

## License

This project is available for educational and research purposes.

## Acknowledgments

This project demonstrates practical implementations of modern information retrieval techniques, combining classical algorithms like BM25 with neural methods for comprehensive search capabilities.