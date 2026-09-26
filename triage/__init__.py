"""Deterministic pre-op triage pipeline.

Policy lives in code (facts/gates/rules/cite); the only thing an LLM ever sees
is document text (documents.py). See ``triage.pipeline.run`` for the entry point.
"""
