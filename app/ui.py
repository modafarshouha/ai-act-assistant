"""Streamlit front end. Loads the graph once and calls it directly.

    streamlit run app/ui.py
"""

import sys
from pathlib import Path

import streamlit as st

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from app.agent import build_graph, initial_state, run_config  # noqa: E402
from app.config import get_settings  # noqa: E402
from app.index import HybridIndex  # noqa: E402
from app.llm import get_llm  # noqa: E402
from app.state import Message  # noqa: E402

EXAMPLES = [
    "What fine can an SMC face for a prohibited practice with EUR 800 million turnover?",
    "When do the Annex III high-risk obligations apply?",
    "What changed about the high-risk timeline, and what is the penalty for non-compliance?",
]


@st.cache_resource
def load():
    """Index and models take seconds to load. Build once per process."""
    settings = get_settings()
    index = HybridIndex.load(settings.index_dir, settings)
    return build_graph(get_llm(settings), index, settings), index, settings


def render_trace(result) -> None:
    scope = result.get("scope", "?")
    verification = result.get("verification")
    timings = result.get("timings", {})

    left, middle, right = st.columns(3)
    left.metric("Scope", scope.replace("_", " "))
    middle.metric("Grounded", "yes" if verification and verification.grounded else "no")
    right.metric("Total", f"{sum(timings.values()):.0f} ms")

    if verification and not verification.grounded:
        for gap in verification.gaps:
            st.warning(gap.detail)

    subtasks = result.get("subtasks", [])
    if subtasks:
        st.caption("Sub-questions")
        for task in subtasks:
            tool = f" — calls `{task.tool_name}`" if task.needs_tool else ""
            st.markdown(f"{task.subtask_id + 1}. {task.question}{tool}")

    tool_results = [r for r in result.get("tool_results", [])]
    if tool_results:
        st.caption("Tool calls")
        for item in tool_results:
            with st.expander(f"{item.tool} — {'ok' if item.ok else 'failed'}"):
                st.json(item.args)
                st.write(item.summary or item.error)
                for caveat in item.value.get("caveats", []):
                    st.info(caveat)

    contexts = result.get("contexts", [])
    if contexts:
        st.caption("Retrieved passages")
        for position, chunk in enumerate(contexts, start=1):
            label = f"[{position}] {chunk.citation} (score {chunk.rerank_score:.2f})"
            with st.expander(label):
                st.write(chunk.body)

    if timings:
        st.caption("Time per node, ms")
        st.bar_chart(timings, horizontal=True)


st.set_page_config(page_title="AI Act assistant", layout="centered")

try:
    graph, index, settings = load()
except FileNotFoundError as exc:
    st.error(str(exc))
    st.stop()

with st.sidebar:
    st.subheader("System")
    st.write(f"{len(index)} indexed passages")
    st.write(f"Provider: `{settings.llm_provider}`")
    if settings.llm_provider == "dummy":
        st.info(
            "The default provider quotes retrieved passages rather than writing "
            "prose. Start with `docker compose --profile llm up` for real generation."
        )
    if st.button("New conversation"):
        st.session_state["turns"] = []
        st.rerun()

st.title("Ask about the AI Act or the GDPR")
st.session_state.setdefault("turns", [])

if not st.session_state["turns"]:
    st.caption("For example:")
    for example in EXAMPLES:
        st.caption(f"• {example}")

for turn in st.session_state["turns"]:
    with st.chat_message("user"):
        st.write(turn["question"])
    with st.chat_message("assistant"):
        st.write(turn["result"]["answer"])
        citations = turn["result"].get("citations", [])
        if citations:
            st.caption("Sources: " + ", ".join(c.citation for c in citations))
        with st.expander("How this answer was reached"):
            render_trace(turn["result"])

if question := st.chat_input("Ask a compliance question...", max_chars=500):
    history = []
    for turn in st.session_state["turns"][-3:]:
        history.append(Message("user", turn["question"]))
        history.append(Message("assistant", turn["result"]["answer"]))

    with st.spinner("Thinking..."):
        result = graph.invoke(initial_state(question, history, settings), run_config(settings))
    st.session_state["turns"].append({"question": question, "result": result})
    st.rerun()
