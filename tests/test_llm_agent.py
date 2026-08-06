import importlib


def test_llm_agent_module_imports_with_runtime_type_annotations():
    module = importlib.import_module("knowledge_engine.llm_agent")

    assert callable(module.PathwayLLMAgent.evaluate_batch)
