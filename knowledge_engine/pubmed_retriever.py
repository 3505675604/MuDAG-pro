"""
MuDAG-Pro PubMed literature-retrieval operator D_PubMed (Section 3.3 of the paper).

Given disease context M_context and pathway-node pair (j, k) to validate,
retrieves literature evidence of directional regulation between pathways from PubMed.

Uses the NCBI E-utilities API (ESearch + EFetch) for retrieval.
"""
import urllib.parse
import urllib.request
import xml.etree.ElementTree as ET
import time
import json
from typing import List, Dict, Tuple, Optional


class PubMedRetriever:
    """
    PubMed literature-retrieval operator D_PubMed (Section 3.3 of the paper).

    Automatically searches PubMed using a specific disease context M_context
    and pathway nodes (j, k) to validate, returning summaries of literature evidence.

    Retrieval constraint defined in Section 3.3 of the paper:
    Literature retrieved by D_PubMed must provide evidence that
    "node j directionally regulates node k."
    """
    def __init__(self, email: str = "researcher@example.com", api_key: Optional[str] = None):
        """
        Args:
            email: Required NCBI Entrez parameter (identifies the user).
            api_key: NCBI API key (raises the rate limit to 10 requests/second).
        """
        self.email = email
        self.api_key = api_key
        self.esearch_url = "https://eutils.ncbi.nlm.nih.gov/entrez/eutils/esearch.fcgi"
        self.efetch_url = "https://eutils.ncbi.nlm.nih.gov/entrez/eutils/efetch.fcgi"

    def search_evidence(
        self,
        source_pathway: str,
        target_pathway: str,
        context_terms: List[str] = None,
        max_results: int = 5
    ) -> List[Dict[str, str]]:
        """
        Retrieve PubMed literature providing evidence that pathway j directionally regulates pathway k (Section 3.3 of the paper).

        Args:
            source_pathway: Name of source pathway j.
            target_pathway: Name of target pathway k.
            context_terms: Disease-context qualifiers.
                Default: ["HR+/HER2- Breast Cancer", "Endocrine Resistance"].
            max_results: Maximum number of articles to return.

        Returns:
            articles: [{"pmid": "12345", "title": "...", "abstract": "..."}, ...]
        """
        if context_terms is None:
            context_terms = ["HR+/HER2- Breast Cancer", "Endocrine Resistance"]

        # Build the search query expression (constrained retrieval in Section 3.3 of the paper).
        context_query = " AND ".join([f'"{term}"' for term in context_terms])

        # Add regulatory-relation keywords to improve precision.
        regulation_query = (
            '(regulation[Title/Abstract] OR signaling[Title/Abstract] '
            'OR pathway[Title/Abstract] OR crosstalk[Title/Abstract] '
            'OR activates[Title/Abstract] OR inhibits[Title/Abstract])'
        )

        query_str = (
            f'("{source_pathway}"[All Fields] OR {source_pathway}[All Fields]) '
            f'AND ("{target_pathway}"[All Fields] OR {target_pathway}[All Fields]) '
            f'AND ({context_query}) '
            f'AND {regulation_query} '
            f'AND (humans[Filter] AND english[Filter])'
        )

        # 1. Run ESearch to obtain the PMID list.
        params = {
            "db": "pubmed",
            "term": query_str,
            "retmode": "json",
            "retmax": str(max_results * 2),  # Fetch extra records for subsequent filtering.
            "sort": "relevance",
            "email": self.email,
        }
        if self.api_key:
            params["api_key"] = self.api_key

        url = f"{self.esearch_url}?{urllib.parse.urlencode(params)}"
        try:
            req = urllib.request.Request(url, headers={'User-Agent': 'Mozilla/5.0'})
            with urllib.request.urlopen(req, timeout=30) as response:
                data = json.loads(response.read().decode('utf-8'))
                pmid_list = data.get("esearchresult", {}).get("idlist", [])
        except Exception as e:
            print(f"[PubMed] ESearch 失败: {e}")
            return []

        if not pmid_list:
            print(f"[PubMed] 未找到 '{source_pathway}' -> '{target_pathway}' 相关文献")
            return []

        # Pause briefly to avoid overly frequent requests (NCBI rate limit: 3 requests/second without an API key).
        time.sleep(0.34)

        # 2. Run EFetch to retrieve article abstracts.
        fetch_params = {
            "db": "pubmed",
            "id": ",".join(pmid_list),
            "retmode": "xml",
            "email": self.email,
        }
        if self.api_key:
            fetch_params["api_key"] = self.api_key

        fetch_url = f"{self.efetch_url}?{urllib.parse.urlencode(fetch_params)}"
        articles = []

        try:
            req = urllib.request.Request(fetch_url, headers={'User-Agent': 'Mozilla/5.0'})
            with urllib.request.urlopen(req, timeout=60) as response:
                xml_data = response.read()
                root = ET.fromstring(xml_data)

                for article_node in root.findall(".//PubmedArticle"):
                    pmid_node = article_node.find(".//PMID")
                    pmid = pmid_node.text if pmid_node is not None else ""

                    title_node = article_node.find(".//ArticleTitle")
                    title = title_node.text if title_node is not None else ""

                    abstract_texts = article_node.findall(".//AbstractText")
                    abstract = " ".join([
                        elem.text for elem in abstract_texts if elem.text
                    ])

                    # Obtain journal and publication-year information.
                    journal_node = article_node.find(".//Journal/Title")
                    journal = journal_node.text if journal_node is not None else ""

                    year_node = article_node.find(".//PubDate/Year")
                    year = year_node.text if year_node is not None else ""

                    if pmid and abstract:
                        articles.append({
                            "pmid": pmid,
                            "title": title.strip(),
                            "abstract": abstract.strip(),
                            "journal": journal.strip(),
                            "year": year.strip(),
                        })

        except Exception as e:
            print(f"[PubMed] EFetch 失败: {e}")

        # Truncate to max_results.
        articles = articles[:max_results]

        if articles:
            print(f"[PubMed] '{source_pathway}' -> '{target_pathway}': "
                  f"检索到 {len(articles)} 篇文献")

        return articles

    def search_hypothetical_edge(
        self,
        source_pathway: str,
        target_pathway: str,
        source_genes: List[str],
        target_genes: List[str],
        disease_context: str = "HR+/HER2- breast cancer endocrine resistance",
        max_results: int = 5,
    ) -> List[Dict[str, str]]:
        """
        Search for literature evidence supporting hypothetical edges across functional branches (Section 3.3 of the paper).

        For cross-branch edges outside the Reactome hierarchy, use gene-pathway associations
        to build a more flexible query that can discover potential cross-pathway regulatory relationships.

        Args:
            source_pathway: Source pathway.
            target_pathway: Target pathway.
            source_genes: List of key genes in the source pathway.
            target_genes: List of key genes in the target pathway.
            disease_context: Disease context.
            max_results: Maximum number of results.

        Returns:
            articles: List of articles.
        """
        # Select representative genes to build the query.
        src_sample = source_genes[:min(5, len(source_genes))]
        tgt_sample = target_genes[:min(5, len(target_genes))]
        gene_terms = " OR ".join([f'"{g}"[All Fields]' for g in src_sample + tgt_sample])

        query_str = (
            f'({gene_terms}) AND '
            f'("{disease_context}"[All Fields]) AND '
            f'(signaling[Title/Abstract] OR regulation[Title/Abstract] OR '
            f'pathway crosstalk[Title/Abstract] OR interaction[Title/Abstract])'
        )

        params = {
            "db": "pubmed",
            "term": query_str,
            "retmode": "json",
            "retmax": str(max_results * 2),
            "sort": "relevance",
            "email": self.email,
        }
        if self.api_key:
            params["api_key"] = self.api_key

        url = f"{self.esearch_url}?{urllib.parse.urlencode(params)}"
        try:
            req = urllib.request.Request(url, headers={'User-Agent': 'Mozilla/5.0'})
            with urllib.request.urlopen(req, timeout=30) as response:
                data = json.loads(response.read().decode('utf-8'))
                pmid_list = data.get("esearchresult", {}).get("idlist", [])
        except Exception as e:
            print(f"[PubMed] 假想边检索失败: {e}")
            return []

        if not pmid_list:
            return []

        time.sleep(0.34)

        fetch_params = {
            "db": "pubmed",
            "id": ",".join(pmid_list),
            "retmode": "xml",
            "email": self.email,
        }
        if self.api_key:
            fetch_params["api_key"] = self.api_key

        fetch_url = f"{self.efetch_url}?{urllib.parse.urlencode(fetch_params)}"
        articles = []

        try:
            req = urllib.request.Request(fetch_url, headers={'User-Agent': 'Mozilla/5.0'})
            with urllib.request.urlopen(req, timeout=60) as response:
                xml_data = response.read()
                root = ET.fromstring(xml_data)

                for article_node in root.findall(".//PubmedArticle"):
                    pmid_node = article_node.find(".//PMID")
                    pmid = pmid_node.text if pmid_node is not None else ""

                    title_node = article_node.find(".//ArticleTitle")
                    title = title_node.text if title_node is not None else ""

                    abstract_texts = article_node.findall(".//AbstractText")
                    abstract = " ".join([e.text for e in abstract_texts if e.text])

                    if pmid and abstract:
                        articles.append({
                            "pmid": pmid,
                            "title": title.strip(),
                            "abstract": abstract.strip(),
                        })

        except Exception as e:
            print(f"[PubMed] 假想边 EFetch 失败: {e}")

        return articles[:max_results]


def build_pubmed_query_for_edge(
    source_pathway: str,
    target_pathway: str,
    disease: str = "HR+/HER2- breast cancer",
    phenotype: str = "endocrine resistance",
) -> str:
    """
    Build a standardized PubMed search-query string (M_context in Section 3.3 of the paper).

    Used to inspect and debug the search expression in the command line or logs.
    """
    query = (
        f'("{source_pathway}"[All Fields] OR {source_pathway}[All Fields]) '
        f'AND ("{target_pathway}"[All Fields] OR {target_pathway}[All Fields]) '
        f'AND ("{disease}"[All Fields] OR breast cancer[MeSH Terms]) '
        f'AND ({phenotype}[Title/Abstract] OR drug resistance[Title/Abstract] '
        f'OR signaling pathway[MeSH Terms]) '
        f'AND (humans[Filter] AND english[Filter])'
    )
    return query
