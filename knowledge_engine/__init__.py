"""
MuDAG-Pro knowledge engine module (Module 1)

Offline LLM + RAG knowledge refinement engine responsible for:
- PubMed literature retrieval (pubmed_retriever.py: PubMedRetriever)
- LLM-based literature parsing and structured rule generation (llm_agent.py: PathwayLLMAgent)
- DAG edge validation, cycle detection, and rule filtering (dag_refine_filter.py: RuleHandbookGenerator)
- Generating the globally frozen rule handbook R_1 (rules_handbook.json, containing 34 cross-branch regulatory edges)
"""
