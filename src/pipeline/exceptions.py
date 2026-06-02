class PipelineError(Exception):
    """Base pipeline failure."""


class NoLabelDetected(PipelineError):
    """No archival label detection on sheet."""


class LlmParseError(PipelineError):
    """LLM response is not valid JSON."""
