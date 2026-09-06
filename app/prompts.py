"""Prompt templates. The task markers are how DummyLLM tells them apart."""

TASK_DECOMPOSE = "decompose"
TASK_SYNTHESIZE = "synthesize"

DECOMPOSE_SYSTEM = (
    "You split compliance questions into independent sub-questions that can each "
    "be researched separately. Output one sub-question per line, no numbering, no "
    "commentary. If the question is already single, output it unchanged."
)

SYNTHESIS_SYSTEM = """You are a compliance research assistant for the EU AI Act and the GDPR. \
You answer only from the numbered passages and computed values supplied to you.

Rules:
- Cite every factual claim with the bracketed number of its passage, like [1].
- Use computed figures and dates exactly as given. Never recalculate them.
- If the passages do not answer the question, say so plainly and stop.
- Quote amounts and dates in the form the regulation uses.
- You are describing what the law says. Do not give legal advice."""


def decompose_prompt(question: str) -> str:
    return (
        f"### TASK: {TASK_DECOMPOSE}\n\n"
        "Split the following compliance question into the smallest set of "
        "independent sub-questions. Most questions need only one. Split only when "
        "the parts would be answered by different provisions.\n\n"
        f"Question: {question}\n\nSub-questions:"
    )


def synthesis_prompt(question: str, context_block: str, tool_block: str = "") -> str:
    parts = [f"### TASK: {TASK_SYNTHESIZE}\n"]
    if tool_block:
        parts.append(
            "Computed values. These are authoritative. Use them exactly as written "
            f"and do not recalculate:\n{tool_block}\n"
        )
    parts.append(
        f"Passages from the regulations:\n{context_block or '(no passages were retrieved)'}\n"
    )
    parts.append(f"Question: {question}\n")
    parts.append("Answer, citing each claim with its passage number:")
    return "\n".join(parts)


def task_of(prompt: str) -> str:
    first = prompt.lstrip().splitlines()[0] if prompt.strip() else ""
    return first.removeprefix("### TASK:").strip() if first.startswith("### TASK:") else ""
