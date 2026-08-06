"""
MuDAG-Pro agent for RAG- and LLM-based literature validation and parameter quantification (Section 3.3 of the paper).

Parses PubMed abstracts and extracts pathway regulatory-rule tuples:
    r = ((j, k), δ, PMID, c)

Where:
    j, k: Upstream and downstream pathways.
    s_ejk ∈ {-1, 0, 1}: -1 inhibition, 0 no significant regulation, 1 activation.
    δ ∈ (0, +∞): Adjustment multiplier.
    PMID: Source literature ID.
    c: Normalized confidence score (assigned automatically by the LLM based on the strength of literature evidence).
"""
import json
import os
import time
from typing import List, Dict, Optional, Tuple, Union


class PathwayLLMAgent:
    """
    Agent for RAG- and LLM-based literature validation and parameter quantification (Section 3.3 of the paper).

    Parses PubMed-retrieved abstracts into structured data
    and outputs structured rule tuples with PMID provenance.

    Agent-based literature validation and parameter quantification from Section 3.3 of the paper:
        s_ejk = Agent(D_PubMed, C_Reactome, M_context, e_jk)
             → r = ((j, k), δ, PMID, c)
    """
    def __init__(self, api_key: Optional[str] = None, model_name: str = "gpt-4o"):
        """
        Args:
            api_key: OpenAI API key (read from the OPENAI_API_KEY environment variable by default).
            model_name: LLM model name (gpt-4o, gpt-4-turbo, or gpt-4).
        """
        self.api_key = api_key or os.getenv("OPENAI_API_KEY")
        self.model_name = model_name

        try:
            from openai import OpenAI
            self.client = OpenAI(api_key=self.api_key)
        except ImportError:
            print("[LLM Agent] Warning: openai 库未安装，将无法调用 LLM")
            self.client = None

    def _build_prompt(
        self,
        source_pathway: str,
        target_pathway: str,
        gene: str,
        articles: List[Dict[str, str]]
    ) -> str:
        """
        Build the standard prompt template used to extract rules (Section 3.3 of the paper).

        Args:
            source_pathway: Source pathway j.
            target_pathway: Target pathway k.
            gene: Trigger gene (for example, PIK3CA or ESR1).
            articles: List of PubMed-retrieved abstracts.
        """
        articles_text = "\n\n".join([
            f"PMID: {a['pmid']}\nTitle: {a.get('title', '')}\nAbstract: {a.get('abstract', '')}"
            for a in articles
        ])

        prompt = f"""You are a precision oncology biological knowledge extraction agent.
Context: HR+/HER2- Breast Cancer, Endocrine Resistance.
Driver Gene Mutation: {gene}
Query Pair: Source Pathway [{source_pathway}] -> Target Pathway [{target_pathway}]

Retrieved Medical Literature Evidence:
{articles_text}

Task:
Analyze if the mutation in {gene} or activation of [{source_pathway}] regulates [{target_pathway}] in HR+/HER2- breast cancer endocrine resistance.

Consider:
1. Does the literature provide direct experimental evidence of regulation?
2. Is the regulation activating or inhibitory?
3. What is the strength/conservativeness of the evidence?
4. Which PMID provides the primary supporting evidence?

Confidence Scoring Guidelines:
- 0.9-1.0: Direct experimental validation (e.g., knockdown/overexpression, reporter assays) in breast cancer
- 0.7-0.89: Strong indirect evidence (e.g., known mechanism in related cancer, multiple corroborating studies)
- 0.5-0.69: Moderate evidence (e.g., correlation in expression data, co-occurrence patterns)
- 0.3-0.49: Weak/suggestive evidence (e.g., bioinformatics inference only)
- 0.0-0.29: Mentioned but no supporting data (do not generate a rule)

Delta (δ) Adjustment Factor Guidelines:
- For strong activating mutations/signaling: δ = 1.5-2.0
- For moderate activating effects: δ = 1.2-1.5
- For weak activating effects: δ = 1.0-1.2
- For weak inhibitory effects: δ = 0.5-1.0
- For moderate inhibitory effects: δ = 0.3-0.5
- For strong inhibitory effects: δ = 0.1-0.3

Return STRICT JSON format matching this schema:
{{
    "source": "{source_pathway}",
    "target": "{target_pathway}",
    "gene": "{gene}",
    "regulation_type": "activation" | "inhibition" | "none",
    "delta": float (positive magnitude of edge modulation, e.g. 0.5 to 2.0. If none, 0.0),
    "pmid": string (the primary PMID supporting this),
    "confidence": float (between 0.0 and 1.0, based on experimental evidence level)
}}
"""
        return prompt

    def evaluate_rule(
        self,
        source_pathway: str,
        target_pathway: str,
        gene: str,
        articles: List[Dict[str, str]]
    ) -> Optional[Dict]:
        """
        Call the LLM agent to generate a structured rule with literature provenance (PMID) (Equation 3.3 in the paper).

        Args:
            source_pathway: Source pathway j.
            target_pathway: Target pathway k.
            gene: Trigger gene.
            articles: List of PubMed abstracts.

        Returns:
            rule: Structured rule dictionary, or None when there is no significant regulatory evidence.
                {
                    "source": "Pathway_J",
                    "target": "Pathway_K",
                    "gene": "PIK3CA",
                    "regulation_type": "activation",
                    "delta": 1.5,
                    "pmid": "12345678",
                    "confidence": 0.85
                }
        """
        if not articles or self.client is None:
            return None

        prompt = self._build_prompt(source_pathway, target_pathway, gene, articles)

        try:
            response = self.client.chat.completions.create(
                model=self.model_name,
                messages=[
                    {"role": "system", "content": "You output JSON only. No markdown, no explanation."},
                    {"role": "user", "content": prompt},
                ],
                temperature=0.0,
                response_format={"type": "json_object"},
            )
            result = json.loads(response.choices[0].message.content)

            reg_type = result.get("regulation_type", "none")
            delta = float(result.get("delta", 0.0))
            confidence = float(result.get("confidence", 0.0))

            # Filter non-informative results.
            if reg_type == "none" or delta <= 0.0:
                return None

            if confidence < 0.3:  # Paper constraint: discard rules with confidence below 0.3.
                return None

            # Sign convention: activation has positive delta (increases edge weight), inhibition has negative delta (decreases edge weight).
            # Section 3.3 of the paper: δ ∈ (0, +∞) is an adjustment multiplier.
            # In practice, activation makes delta > 1 and inhibition makes delta < 1.
            # Keep delta as an absolute value here and record the direction in regulation_type.
            final_delta = delta if reg_type == "activation" else -delta

            rule = {
                "source": source_pathway,
                "target": target_pathway,
                "gene": gene,
                "delta": final_delta,
                "pmid": str(result.get("pmid", articles[0]["pmid"])),
                "confidence": min(max(confidence, 0.0), 1.0),
            }
            return rule

        except Exception as e:
            print(f"[LLM Agent] 规则解析失败 ({source_pathway} -> {target_pathway}): {e}")
            return None

    def evaluate_batch(
        self,
        edge_gene_pairs: List[Dict],
        article_cache: Dict[Tuple[str, str], List[Dict[str, str]]],
    ) -> List[Dict]:
        """
        Evaluate regulatory rules for multiple pathway-gene pairs in a batch (Section 3.3 of the paper).

        Args:
            edge_gene_pairs: [{"source": ..., "target": ..., "gene": ...}, ...]
            article_cache: Cache dictionary mapping (source, target) to articles.

        Returns:
            rules: List of all valid structured rules.
        """
        all_rules = []
        n_total = len(edge_gene_pairs)

        for i, pair in enumerate(edge_gene_pairs):
            src = pair["source"]
            tgt = pair["target"]
            gene = pair["gene"]

            # Retrieve articles from the cache.
            articles = article_cache.get((src, tgt), [])

            if not articles:
                continue

            if (i + 1) % 10 == 0:
                print(f"[LLM Agent] 进度: {i + 1}/{n_total}")

            rule = self.evaluate_rule(src, tgt, gene, articles)
            if rule is not None:
                all_rules.append(rule)

            # API rate limiting.
            if (i + 1) % 3 == 0:
                time.sleep(0.5)

        print(f"[LLM Agent] 批量评估完成: {len(all_rules)}/{n_total} 条有效规则")
        return all_rules


def load_prompt_template(config_path: str) -> Dict:
    """
    Load the LLM prompt configuration template (config/llm_prompt_config.json).

    This can customize the prompt template and override the default system prompt.
    """
    if not os.path.exists(config_path):
        return {}
    with open(config_path, 'r', encoding='utf-8') as f:
        config = json.load(f)
    return config
