# NLP Evaluation Task: Economic News Aggregator

This project implements an NLP pipeline in a single Jupyter Notebook (`NLP_evaluation.ipynb`) for extracting entities, summarizing documents, and designing an AI agent (*EconoBot*) to provide economic insights from news articles. It covers three parts: data preparation/exploration, information extraction/summarization, and agentic system design, using the NLTK Reuters Corpus.

## Setup Instructions

### Prerequisites
- Python 3.11
- Jupyter Notebook or Google Colab
- Sufficient disk space for models (~5 GB for Transformers, spaCy)

### Installation
1. Clone the repository (or create a project directory):
   ```bash
   git clone https://github.com/habeshaethiopia/NLP-evaluation
   cd NLP-evaluation
   ```
2. Install dependencies from `requirements.txt`:
   ```bash
   pip install -r requirements.txt
   ```
3. Download spaCy’s English model:
   ```bash
   python -m spacy download en_core_web_sm
   ```
4. Download NLTK data:
   ```python
   import nltk
   nltk.download('reuters')
   nltk.download('punkt')
   nltk.download('stopwords')
   ```
5. Launch Jupyter Notebook:
   ```bash
   jupyter notebook NLP_evaluation.ipynb
   ```
   Alternatively, upload `NLP_evaluation.ipynb` to Google Colab.

### Dependencies
Listed in `requirements.txt`:
- `nltk==3.8.1`: For Reuters Corpus and preprocessing.
- `spacy==3.7.4`: For tokenization, lemmatization, and NER.
- `transformers==4.38.2`: For BART summarization.
- `sentence-transformers==2.5.1`: For document selection.
- `rouge-score==0.1.2`: For ROUGE evaluation.
- `pandas==2.2.1`, `matplotlib==3.8.3`, `seaborn==0.13.2`: For EDA and visualization.

## Dataset Source and Preprocessing Description

### Dataset
- **Source**: NLTK Reuters Corpus
  - Contains ~10,788 news articles on economic topics (e.g., trade, finance).
  - Accessed via `nltk.corpus.reuters` in the notebook.
  - No external download required; included in NLTK.
- **Structure**: Unstructured text documents, varying in length, with no ground truth summaries.

### Preprocessing
- **Steps** (Part 1, notebook section “Data Preparation & Exploration”):
  - **Tokenization**: Split text into tokens using spaCy.
  - **Lowercasing**: Convert text to lowercase for consistency.
  - **Stop-word Removal**: Remove common words (e.g., “the”, “and”) using NLTK’s stopword list.
  - **Lemmatization**: Reduce words to their base form (e.g., “running” → “run”) using spaCy.
- **Implementation**: Applied to a subset (e.g., 1000 documents) for efficiency.
- **Output**: Tokenized, lemmatized documents for EDA and downstream tasks, stored as lists in the notebook.

## How to Run Each Module

The `NLP_evaluation.ipynb` notebook is organized into sections corresponding to Parts 1–3 and evaluation. Run cells sequentially in Jupyter Notebook or Google Colab, ensuring dependencies are installed.

### Part 1: Data Preparation & Exploration
- **Notebook Section**: “Data Preparation & Exploration”
- **Purpose**: Load Reuters Corpus, preprocess documents, and perform exploratory data analysis (EDA) on document lengths, frequent words, and entity frequencies.
- **How to Run**:
  1. Open `NLP_evaluation.ipynb` in Jupyter or Colab.
  2. Run cells under “Data Preparation & Exploration”.
  3. Ensure NLTK’s Reuters Corpus and spaCy’s `en_core_web_sm` are installed.
- **Output**:
  - Visualizations (histograms/bar charts) for document lengths, top words, and entity types (displayed inline or saved as PNGs).
  - Console output of EDA statistics (e.g., word frequencies).
- **Notes**: Adjust `n_documents` (e.g., 1000) to process more/fewer documents. Requires `matplotlib` and `seaborn`.

### Part 2: Information Extraction & Summarization
- **Notebook Section**: “Information Extraction & Summarization”
- **Purpose**: Extract entities (rule-based for dates/metrics, spaCy for NER) and generate abstractive summaries using Hugging Face’s `facebook/bart-large-cnn`.
- **How to Run**:
  1. Run cells under “Information Extraction & Summarization”.
  2. Ensure `transformers` and `spacy` are installed.
- **Output**:
  - `extracted_entities.json`: JSON file with entities (e.g., “4.5 billion baht”, “Thailand”).
  - Console output of summaries and qualitative examples (e.g., Document 2: “Survey of 19 provinces... 7-12 pct of China’s grain stocks...”).
  - Note: Document 1’s summarization may fail (“index out of range” error).
- **Notes**:
  - Debug Document 1 error by truncating input to 1024 tokens (see below).
  - Entity extraction missed some metrics (e.g., “4.5 billion baht”) due to restrictive regex patterns.

### Part 2 Evaluation
- **Notebook Section**: “Summary Evaluation”
- **Purpose**: Compute ROUGE (ROUGE-1, ROUGE-2, ROUGE-L) and BLEU scores for summaries against reference summaries.
- **How to Run**:
  1. Run cells under “Summary Evaluation”.
  2. Provide reference summaries in the notebook (e.g., as a list) or a separate file.
- **Output**:
  - Console output of scores (e.g., ROUGE-1 F1: 0.5275, BLEU: 0.0656).
  - Warnings for BLEU = 0 (Documents 2, 3) due to no 3-/4-gram overlaps.
- **Notes**:
  - Low scores indicate phrasing differences (e.g., “pct” vs. “%”). Use `SmoothingFunction` for BLEU to avoid zeros.
  - Provide ground truth summaries for accurate evaluation.

### Part 3: Agentic System Design
- **Notebook Section**: “Agentic System Design (News Aggregator Agent)”
- **Purpose**: Implement *News Aggregator Agent*, which answers economic queries (e.g., “Summarize today’s news about trade policies”) by selecting documents, extracting entities, summarizing, and synthesizing responses.
- **How to Run**:
  1. Run cells under “Agentic System Design (News Aggregator Agent)”.
  2. Input a query via the `news_aggregator_agent` function (e.g., `news_aggregator_agent("Summarize today’s news about trade policies")`).
- **Output**:
  - Console output of the synthesized response, including summaries and key entities.
  - `knowledge_base.json`: Stores processed entities and summaries.


## Explanation of Agent Design

### Scenario
- **Agent**: *News Aggregator Agent*
- **Purpose**: Continuously monitors news sources (e.g., Reuters Corpus, RSS feeds, NewsAPI.org) to produce concise summaries of breaking economic news, either on a schedule (e.g., daily digests) or in response to user queries (e.g., “Summarize today’s news about trade policies”). It tackles information overload by delivering timely, curated summaries.
- **Problem**: Users need quick insights from large volumes of news without manually sifting through redundant or irrelevant articles.
- **Example Query**: “Summarize today’s news about climate change” or “What are recent trade policies affecting Asian economies?”

### Architecture
*News Aggregator Agent* uses a modular pipeline to process news and deliver insights:
1. **Query Parser** (spaCy):
   - Parses user queries or scheduled tasks to extract keywords (e.g., “trade policies”, “climate change”) and entities (e.g., “Asia” as GPE) using spaCy’s NLP capabilities.
2. **News Retrieval Module** (NewsAPI.org, RSS, or sentence-transformers):
   - Fetches relevant articles using keyword-based search or semantic similarity (via `all-MiniLM-L6-v2` embeddings). For Reuters Corpus, uses vector search to select top 5 documents.
3. **Content Extraction Module** (regex + spaCy):
   - Rule-based: Extracts dates (e.g., “15 March 2025”) and metrics (e.g., “4.5 billion baht”) using regex.
   - NER: Extracts PERSON, ORGANIZATION, GPE, DATE entities using spaCy.
   - Optionally, identifies salient sentences by ranking based on relevance to the query.
4. **Summarization Module** (BART):
   - Generates abstractive summaries (max 100 words) for each article using `facebook/bart-large-cnn`, condensing extracted content into concise text.
5. **Filtering/Indexing Module** (optional, FAISS):
   - Uses a vector database to store article embeddings and metadata, filtering out duplicates or previously seen stories.
6. **Reasoning Module** (custom or Grok 3):
   - Merges summaries and entities into a cohesive report, ranking by relevance and ensuring no redundant information. Optionally uses an LLM for advanced synthesis.
7. **Memory Component** (JSON or FAISS):
   - Stores past articles, summaries, and user preferences (e.g., favorite topics) to avoid re-summarizing known stories and personalize responses.

### Decision Logic and Tool Chaining
- **Query Parsing**: Extracts topics or keywords (e.g., “climate change”) to guide retrieval. For broad queries (e.g., “today’s news”), identifies trending topics via pre-search.
- **Document Retrieval**: Uses NewsAPI.org or vector search (Reuters Corpus) to fetch top 5–10 relevant articles. Clusters articles to reduce duplication (e.g., grouping similar stories).
- **Extraction and Summarization**:
  - Extracts key facts or sentences (e.g., “Who? What?”) from each article using NER or sentence ranking.
  - Summarizes extracted content into a concise paragraph per article.
  - Merges summaries, ranking by relevance or combining into a single report.
- **Answer Formation**: Compiles a natural language response (e.g., “Recent trade policies include U.S. tariffs on Japan...”) with optional entity tables. Performs consistency checks to ensure query coverage.
- **Memory**: Logs article metadata (e.g., IDs, embeddings) and user preferences in `knowledge_base.json` or FAISS, enabling continuity (e.g., “Since last week’s trade query...”).

### Example Workflow
- **Query**: “Summarize today’s news about trade policies.”
- **Process**:
  1. Parse: Extract “trade policies” as topic.
  2. Retrieve: Fetch top 5 articles via vector search (Reuters) or NewsAPI.org.
  3. Extract: Identify entities (e.g., “Thailand”, “4.5 billion baht”) and key sentences.
  4. Summarize: Generate summaries (e.g., “Thailand’s trade deficit widened to 4.5 billion baht...”).
  5. Synthesize: “Recent trade policies include U.S.-Japan tariff disputes and Thailand’s Q1 1987 deficit increase...”
- **Output**: Concise report with summaries and entities, stored in `knowledge_base.json`.

### Limitations
- **Summarization Error**: Document 1’s “index out of range” error (Part 2) requires stricter input truncation (e.g., 1024 tokens) to prevent crashes.
- **Evaluation**: Low ROUGE/BLEU scores (ROUGE-1 F1: 0.5275, BLEU: 0.0656) due to phrasing differences (e.g., “pct” vs. “%”). Standardize terms or use BLEU smoothing.
- **Scalability**: For real-time news, integrate NewsAPI.org or RSS feeds. FAISS can enhance document retrieval efficiency.

