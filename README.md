# AGENTIC RAG_OLAMA

## Background

The goal of this system is to build a **fully local, CPU-only, agentic RAG architecture** using Ollama-hosted open-source LLMs. The system supports **multiple knowledge domains** (Tour Guide and Indian Constitution Law), each with its own retrieval pipeline, and uses an LLM-based planner to dynamically decide which retriever(s) to invoke. The system emphasizes correctness through grounding, metadata enrichment, hybrid retrieval, reranking, and self-reflective corrective loops.

---

## Requirements

### Must Have

* Domain-specific RAG pipelines (Tour Guide, Law)
* Local-only execution (no paid APIs)
* FAISS-based vector storage with persistence
* Hybrid retrieval (BM25 + Vector)
* Metadata enrichment and filtering
* MMR-based diversity selection
* Reranking for relevance
* Grounded answer generation
* Self-reflection and corrective iteration (max 1 loop)

### Should Have

* Modular, production-ready repository structure
* Config-driven behavior (CPU tuning, k values)
* Clear separation of ingestion, retrieval, agents

### Could Have

* Additional domains
* LangGraph visualization

### Won’t Have (for MVP)

* Cloud-hosted LLMs
* GPU dependency
* Neo4j / GraphRAG

---

## Method

### High-Level Architecture

```
User Query
   ↓
Planner Agent (LLM)
   ↓
Domain Decision
   ├── Tour Guide Retriever Tool
   ├── Law Retriever Tool
   ↓
Hybrid Retrieval (BM25 + FAISS)
   ↓
MMR Selection
   ↓
Reranker
   ↓
Answer Agent (LLM)
   ↓
Reflection Agent
   ↓
Final Answer
```

Each domain has an independent FAISS index and BM25 corpus. The planner agent decides which retriever(s) to invoke. Retrieved context is passed to the answer agent under strict grounding rules. A reflection agent validates answer quality and may trigger one corrective retrieval.

---

## Concrete Folder Structure (Production-Ready)

```
rag_agentic_dual_domain/
├── README.md
├── pyproject.toml
├── config/
│   ├── models.yaml
│   ├── retrieval.yaml
│   └── cpu.yaml
│
├── data/
│   ├── raw_docs/
│   │   ├── tour_guide.pdf
│   │   └── indian_constitution.pdf
│   │
│   ├── processed/
│   │   ├── tour_guide_chunks.json
│   │   └── law_chunks.json
│   │
│   └── faiss/
│       ├── tour_guide/
│       │   ├── index.faiss
│       │   └── index.pkl
│       └── law/
│           ├── index.faiss
│           └── index.pkl
│
├── ingestion/
│   ├── loader.py            # unstructured.io PDF parsing
│   ├── normalizer.py        # text normalization
│   ├── chunker.py           # recursive chunk splitter
│   └── enrich.py            # metadata enrichment
│
├── embeddings/
│   ├── embedder.py          # BGE embeddings
│   └── indexer.py           # FAISS build/save/load
│
├── retrieval/
│   ├── bm25.py              # keyword search
│   ├── vector.py            # FAISS search
│   ├── hybrid.py            # BM25 + vector merge
│   ├── mmr.py               # diversity selection
│   └── reranker.py          # relevance reranking
│
├── agents/
│   ├── planner.py           # domain decision agent
│   ├── tour_tool.py         # tour guide retriever tool
│   ├── law_tool.py          # law retriever tool
│   ├── answer.py            # grounded answer agent
│   └── reflector.py         # self-reflection & correction
│
├── prompts/
│   ├── planner.txt
│   ├── answer.txt
│   └── reflection.txt
│
├── schemas/
│   └── outputs.py           # Pydantic response models
│
├── graph/
│   └── rag_graph.py         # LangGraph orchestration
│
├── eval/
│   └── grounding.py         # hallucination checks
│
└── main.py                  # system entry point
```

---

## Implementation

### Phase 1: Ingestion & Indexing

* Parse PDFs using `unstructured`
* Normalize and chunk content
* Enrich metadata (domain, source, page)
* Generate embeddings and persist FAISS indexes per domain

### Phase 2: Retrieval Layer

* Implement BM25 and FAISS retrievers
* Merge results using hybrid strategy
* Apply MMR and reranker

### Phase 3: Agent Layer

* Planner agent selects domain(s)
* Retriever tools fetch evidence
* Answer agent generates grounded response
* Reflection agent validates and optionally retries

### Phase 4: Orchestration

* Connect agents using LangGraph
* Enforce max reflection loop

---

## Milestones

1. PDF ingestion + FAISS persistence
2. Single-domain RAG query
3. Dual-domain retriever tools
4. Planner-based routing
5. Reflection & corrective RAG

---

## Gathering Results

* Measure answer grounding accuracy
* Validate correct domain routing
* Track latency on CPU-only execution
* Evaluate reflection-trigger frequency

---


