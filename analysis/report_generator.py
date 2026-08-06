"""
MuDAG-Pro LLM white-box clinical decision-report generation module (Section 3.7 of the paper).

Uses key-pathway contribution rankings, somatic mutation context, and matched literature rules
to prompt an LLM to generate personalized prognostic reports readable by physicians.

Workflow:
1. Input: PI score, top K pathways, mutation signature, and matched rules.
2. Assemble structured context.
3. Generate a professional medical report with the LLM.
4. Output: auditable text report + JSON.
"""
import json
import os
from typing import Dict, List, Optional, Tuple
from datetime import datetime


class ReportGenerator:
    """
    Personalized white-box clinical decision-report generator.

    Combines Ridge-Cox β coefficients, SHAP attributions, and mutation-matching rules
    to generate interpretable prognostic reports for oncologists.
    """

    def __init__(
        self,
        pathway_names: List[str],
        rules_handbook: Optional[List[Dict]] = None,
        llm_agent=None,
    ):
        """
        Args:
            pathway_names: List of 331 core pathway names.
            rules_handbook: Global rule handbook R_1.
            llm_agent: LLMAgent instance (optional, used for LLM-enhanced reports).
        """
        self.pathway_names = pathway_names
        self.rules_handbook = rules_handbook or []
        self.llm_agent = llm_agent

    def generate_report(
        self,
        patient_id: str,
        pi_score: float,
        risk_group: str,
        risk_percentile: float,
        top_pathways: List[Dict],
        mutation_signature: List[Tuple[str, str]],
        matched_rules: List[Dict],
        clinical_info: Optional[Dict] = None,
        use_llm: bool = True,
    ) -> Dict:
        """
        Generate a personalized prognostic report.

        Args:
            patient_id: Patient ID.
            pi_score: Prognostic index PI.
            risk_group: Risk group ("High Risk" / "Low Risk").
            risk_percentile: Risk percentile.
            top_pathways: Key driver pathways (including β, HR, and 95% CI).
            mutation_signature: Mutation signature S_i.
            matched_rules: Matched literature rules.
            clinical_info: Optional clinical information {"age": ..., "stage": ..., "grade": ...}.
            use_llm: Whether to generate an enhanced report with the LLM.

        Returns:
            report: {
                "patient_id": ...,
                "timestamp": ...,
                "summary": ...,
                "risk_assessment": ...,
                "molecular_rationale": ...,
                "biological_mechanism": ...,
                "clinical_implications": ...,
                "limitations": ...,
                "raw_data": {...}
            }
        """
        # Build the report context.
        context = self._build_report_context(
            patient_id, pi_score, risk_group, risk_percentile,
            top_pathways, mutation_signature, matched_rules, clinical_info,
        )

        if use_llm and self.llm_agent is not None:
            # LLM-enhanced report.
            report = self._generate_llm_report(context)
        else:
            # Deterministic template-based report.
            report = self._generate_template_report(context)

        report["raw_data"] = context
        report["timestamp"] = datetime.now().isoformat()

        return report

    def _build_report_context(
        self,
        patient_id: str,
        pi_score: float,
        risk_group: str,
        risk_percentile: float,
        top_pathways: List[Dict],
        mutation_signature: List[Tuple[str, str]],
        matched_rules: List[Dict],
        clinical_info: Optional[Dict],
    ) -> Dict:
        """Assemble all context required by the report."""
        context = {
            "patient_id": patient_id,
            "pi_score": round(float(pi_score), 4),
            "risk_group": risk_group,
            "risk_percentile": round(float(risk_percentile), 1),
            "top_pathways": top_pathways,
            "mutation_signature": [
                {"gene": g, "variant": v} for g, v in mutation_signature
            ],
            "matched_rules": matched_rules,
            "n_mutations": len(mutation_signature),
            "n_matched_rules": len(matched_rules),
        }

        if clinical_info:
            context["clinical_info"] = clinical_info

        # Build the risk-driver summary.
        risk_drivers = []
        protective_factors = []
        for pw in top_pathways:
            if pw.get("risk_direction") == "risk":
                risk_drivers.append({
                    "pathway": pw["pathway"],
                    "hr": pw.get("hr", 1.0),
                    "beta": pw.get("beta", 0.0),
                })
            else:
                protective_factors.append({
                    "pathway": pw["pathway"],
                    "hr": pw.get("hr", 1.0),
                    "beta": pw.get("beta", 0.0),
                })

        context["risk_drivers"] = risk_drivers
        context["protective_factors"] = protective_factors

        # Build the mutation-pathway-risk attribution chain.
        attribution_chains = self._build_attribution_chains(
            mutation_signature, matched_rules, top_pathways
        )
        context["attribution_chains"] = attribution_chains

        return context

    def _build_attribution_chains(
        self,
        mutation_signature: List[Tuple[str, str]],
        matched_rules: List[Dict],
        top_pathways: List[Dict],
    ) -> List[Dict]:
        """
        Build the "driver mutation → perturbed pathway → high-weight downstream pathway → accumulated risk" attribution chain
        (Section 3.7 of the paper).
        """
        chains = []

        top_pathway_names = {p["pathway"] for p in top_pathways}

        for rule in matched_rules:
            src = rule.get("source_pathway", "")
            tgt = rule.get("target_pathway", "")

            # Check whether the target pathway is in the key-pathway list.
            if tgt in top_pathway_names:
                matched_genes = rule.get("matched_genes", [])
                for gene in matched_genes:
                    # Find the specific variant for this gene.
                    variant = ""
                    for g, v in mutation_signature:
                        if g == gene:
                            variant = v
                            break

                    chain = {
                        "mutation": f"{gene} {variant}" if variant else gene,
                        "source_pathway": src,
                        "target_pathway": tgt,
                        "regulation_type": rule.get("regulation_type", "modulates"),
                        "delta": rule.get("delta", 1.0),
                        "confidence": rule.get("confidence", 0.5),
                        "pmid": rule.get("pmid", ""),
                    }
                    chains.append(chain)

        return chains

    def _generate_template_report(self, context: Dict) -> Dict:
        """Generate a deterministic template-based report when the LLM is unavailable."""
        risk_drivers = context.get("risk_drivers", [])
        protective = context.get("protective_factors", [])
        chains = context.get("attribution_chains", [])
        mutations = context.get("mutation_signature", [])
        pi = context["pi_score"]
        risk_group = context["risk_group"]
        percentile = context["risk_percentile"]

        # Risk assessment.
        if risk_group == "High Risk":
            risk_text = (
                f"患者 PI = {pi:.4f}，位于队列 {percentile:.1f}% 风险百分位，"
                f"被分类为高风险组。相较于低风险组患者，该患者具有显著更高的"
                f"不良预后事件风险。"
            )
        else:
            risk_text = (
                f"患者 PI = {pi:.4f}，位于队列 {percentile:.1f}% 风险百分位，"
                f"被分类为低风险组。"
            )

        # Molecular mechanism.
        mol_parts = []
        if risk_drivers:
            top_drivers = ", ".join(
                [f"{d['pathway']} (HR={d['hr']:.2f})" for d in risk_drivers[:5]]
            )
            mol_parts.append(f"主要风险驱动通路: {top_drivers}")
        if protective:
            top_protective = ", ".join(
                [f"{p['pathway']} (HR={p['hr']:.2f})" for p in protective[:5]]
            )
            mol_parts.append(f"保护性通路: {top_protective}")

        # Mutation attribution.
        mut_parts = []
        if chains:
            for chain in chains[:5]:
                mut_parts.append(
                    f"{chain['mutation']} → {chain['regulation_type']} "
                    f"{chain['source_pathway']} → {chain['target_pathway']} "
                    f"(PMID: {chain.get('pmid', 'N/A')})"
                )
        elif mutations:
            mut_parts.append(
                f"该患者携带 {len(mutations)} 个功能性突变，"
                f"未匹配到显著的边权调整规则。"
            )

        # Clinical recommendations.
        if risk_group == "High Risk" and risk_drivers:
            clinical_text = (
                "建议密切随访监测，可结合分子肿瘤委员会讨论进一步治疗方案。"
                "PI3K/AKT/mTOR 通路高活性可能提示 CDK4/6 抑制剂联合内分泌治疗获益有限，"
                "可检测 PIK3CA 突变状态以评估 PI3Kα 抑制剂 (如 Alpelisib) 治疗机会。"
            )
        else:
            clinical_text = (
                "患者处于低风险组，建议常规随访监测。"
            )

        # Limitations.
        limitations = (
            "本报告基于 MuDAG-Pro 模型自动生成，仅供临床参考，不应作为独立诊断依据。"
            "模型在 TCGA-BRCA 发现集中训练，外部验证集上的性能见论文 Table 1。"
            "PI 为连续值，风险分组基于队列中位数，不同队列的阈值可能存在差异。"
        )

        report = {
            "patient_id": context["patient_id"],
            "summary": f"预后指数 PI = {pi:.4f}，风险分组: {risk_group}",
            "risk_assessment": risk_text,
            "molecular_rationale": " ".join(mol_parts),
            "biological_mechanism": " ".join(mut_parts) if mut_parts else "未检测到可归因的突变-通路风险机制链。",
            "clinical_implications": clinical_text,
            "limitations": limitations,
        }

        return report

    def _generate_llm_report(self, context: Dict) -> Dict:
        """Use the LLM to generate an enhanced medical report."""
        if self.llm_agent is None:
            return self._generate_template_report(context)

        from knowledge_engine.llm_agent import PathwayLLMAgent

        # Load the prompt configuration.
        prompt_config_path = os.path.join(
            os.path.dirname(os.path.dirname(__file__)),
            "config", "llm_prompt_config.json"
        )
        with open(prompt_config_path, 'r', encoding='utf-8') as f:
            prompt_config = json.load(f)

        report_config = prompt_config.get("report_generation", {})
        system_prompt = report_config.get(
            "system_prompt",
            "You are an expert oncologist and computational pathologist."
        )
        template = report_config.get("generation_prompt", "")

        # Format the top pathways.
        shap_text = "\n".join([
            f"- {p['pathway']}: HR = {p.get('hr', 'N/A'):.3f} "
            f"(95% CI: {p.get('hr_95ci_lower', 'N/A'):.3f}–{p.get('hr_95ci_upper', 'N/A'):.3f}), "
            f"β = {p.get('beta', 'N/A'):.4f}"
            for p in context.get("top_pathways", [])[:10]
        ])

        # Format mutations.
        mut_text = "\n".join([
            f"- {m['gene']} {m.get('variant', '')}"
            for m in context.get("mutation_signature", [])
        ]) or "No mutations detected"

        # Format matched rules.
        rules_text = "\n".join([
            f"- {r.get('source_pathway', '?')} → {r.get('target_pathway', '?')}: "
            f"{r.get('regulation_type', '?')}, "
            f"Δ={r.get('delta', '?'):.2f}, "
            f"c={r.get('confidence', '?'):.2f}, "
            f"PMID: {r.get('pmid', '?')}"
            for r in context.get("matched_rules", [])[:10]
        ]) or "No matched rules"

        user_prompt = template.format(
            patient_id=context["patient_id"],
            pi_score=context["pi_score"],
            risk_group=context["risk_group"],
            risk_percentile=context["risk_percentile"],
            shap_top_pathways=shap_text,
            key_mutations=mut_text,
            matched_rules=rules_text,
        )

        response = self.llm_agent._call_llm(
            system_prompt=system_prompt,
            user_prompt=user_prompt,
        )

        # Attempt to parse the LLM output.
        sections = self._parse_llm_report(response)

        return {
            "patient_id": context["patient_id"],
            "summary": f"PI = {context['pi_score']:.4f}, {context['risk_group']}",
            "risk_assessment": sections.get("Risk Assessment", response[:500]),
            "molecular_rationale": sections.get("Molecular Rationale", ""),
            "biological_mechanism": sections.get("Biological Mechanism", ""),
            "clinical_implications": sections.get("Clinical Implications", ""),
            "limitations": sections.get("Limitations & Caveats", ""),
            "llm_full_response": response,
        }

    def _parse_llm_report(self, response: str) -> Dict[str, str]:
        """Parse the structured sections of an LLM report."""
        sections = {}
        current_section = None
        current_text = []

        # Expected section markers in the LLM output.
        section_markers = [
            "Risk Assessment",
            "Molecular Rationale",
            "Biological Mechanism",
            "Clinical Implications",
            "Limitations",
        ]

        for line in response.split('\n'):
            found_marker = False
            for marker in section_markers:
                if marker.lower() in line.lower() and (
                    line.strip().startswith('#') or
                    line.strip().startswith('**') or
                    marker.lower() in line.lower().split(':', 1)[0]
                ):
                    if current_section:
                        sections[current_section] = '\n'.join(current_text).strip()
                    current_section = marker
                    current_text = [line]
                    found_marker = True
                    break
            if not found_marker:
                if current_section:
                    current_text.append(line)
                else:
                    current_section = "Risk Assessment"
                    current_text = [line]

        if current_section and current_text:
            sections[current_section] = '\n'.join(current_text).strip()

        return sections

    def save_report(
        self,
        report: Dict,
        output_dir: str = "./outputs/reports",
        format: str = "both",
    ) -> str:
        """
        Save a report to files.

        Args:
            report: Report dictionary.
            output_dir: Output directory.
            format: "json", "txt", or "both".

        Returns:
            base_path: Base path for the report files.
        """
        os.makedirs(output_dir, exist_ok=True)
        patient_id = report.get("patient_id", "unknown")
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        base_path = os.path.join(output_dir, f"report_{patient_id}_{timestamp}")

        if format in ("json", "both"):
            with open(f"{base_path}.json", 'w', encoding='utf-8') as f:
                json.dump(report, f, indent=2, ensure_ascii=False)

        if format in ("txt", "both"):
            with open(f"{base_path}.txt", 'w', encoding='utf-8') as f:
                f.write(self._format_report_text(report))

        return base_path

    def _format_report_text(self, report: Dict) -> str:
        """Format a report as plain text."""
        lines = []
        lines.append("=" * 70)
        lines.append("MuDAG-Pro 个体化预后分析报告")
        lines.append("=" * 70)
        lines.append(f"患者 ID: {report.get('patient_id', 'N/A')}")
        lines.append(f"生成时间: {report.get('timestamp', 'N/A')}")
        lines.append("")
        lines.append("-" * 50)
        lines.append("1. 风险摘要")
        lines.append("-" * 50)
        lines.append(report.get("summary", "N/A"))
        lines.append("")
        lines.append("-" * 50)
        lines.append("2. 风险评估")
        lines.append("-" * 50)
        lines.append(report.get("risk_assessment", "N/A"))
        lines.append("")
        lines.append("-" * 50)
        lines.append("3. 分子机制解释")
        lines.append("-" * 50)
        lines.append(report.get("molecular_rationale", "N/A"))
        lines.append("")
        lines.append("-" * 50)
        lines.append("4. 生物学机制归因")
        lines.append("-" * 50)
        lines.append(report.get("biological_mechanism", "N/A"))
        lines.append("")
        lines.append("-" * 50)
        lines.append("5. 临床建议")
        lines.append("-" * 50)
        lines.append(report.get("clinical_implications", "N/A"))
        lines.append("")
        lines.append("-" * 50)
        lines.append("6. 局限性与注意事项")
        lines.append("-" * 50)
        lines.append(report.get("limitations", "N/A"))
        lines.append("")
        lines.append("=" * 70)
        lines.append("本报告由 MuDAG-Pro 自动生成 | 仅供临床参考")

        return "\n".join(lines)

    def generate_batch_reports(
        self,
        patient_results: List[Dict],
        output_dir: str = "./outputs/reports",
        use_llm: bool = False,
    ) -> List[str]:
        """
        Generate patient reports in a batch.

        Args:
            patient_results: List of patient results.
            output_dir: Output directory.
            use_llm: Whether to use the LLM (usually disabled for batches to save cost).

        Returns:
            report_paths: List of report file paths.
        """
        paths = []
        for pr in patient_results:
            report = self.generate_report(
                patient_id=pr["patient_id"],
                pi_score=pr["pi_score"],
                risk_group=pr["risk_group"],
                risk_percentile=pr["risk_percentile"],
                top_pathways=pr.get("top_pathways", []),
                mutation_signature=pr.get("mutation_signature", []),
                matched_rules=pr.get("matched_rules", []),
                clinical_info=pr.get("clinical_info"),
                use_llm=use_llm,
            )
            path = self.save_report(report, output_dir=output_dir)
            paths.append(path)
        return paths
