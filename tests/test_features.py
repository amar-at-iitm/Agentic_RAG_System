
from pipelines.multi_agent_orchestrator import MultiAgentOrchestrator, OrchestratorResult

def test_telemetry_fields_presence():
    orchestrator = MultiAgentOrchestrator()
    # We mock the internal agents to avoid actual calls (and costs/latency)
    # But for now, since we don't have a mocking framework set up deeply, we'll just check the dataclass structure
    # or rely on the fact that we can instantiate the result object.
    
    # Actually, let's just check if the class has the fields we expect, 
    # as running a full query might require the agents to be functional (which they are)
    # but we want a quick unit test.
    
    # Let's inspect the dataclass annotations
    annotations = OrchestratorResult.__annotations__
    assert "latency_seconds" in annotations
    assert "token_usage" in annotations

def test_pii_scrubbing_logic():
    from pipelines.rag_pipeline import RAGPipeline
    
    text = "Contact me at test@example.com or 123-456-7890."
    scrubbed = RAGPipeline.scrub_pii(text)
    
    assert "test@example.com" not in scrubbed
    assert "123-456-7890" not in scrubbed
    assert "[EMAIL_REDACTED]" in scrubbed
    assert "[PHONE_REDACTED]" in scrubbed
    
    safe_text = "This is a safe string."
    assert RAGPipeline.scrub_pii(safe_text) == safe_text

def test_rouge_metric_logic():
    from evaluation.metrics import calculate_rouge
    
    ref = "The quick brown fox jumps over the dog."
    cand = "The quick brown fox jumps over the lazy dog."
    
    scores = calculate_rouge(ref, cand)
    
    assert "rouge1" in scores
    assert "rouge2" in scores
    assert "rougeL" in scores
    assert scores["rouge1"] > 0.0

