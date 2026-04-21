"""
capstone_streamlit.py
Study Buddy — Physics Assistant (Streamlit UI)
Run with: streamlit run capstone_streamlit.py
"""

import streamlit as st
from agent import initialize, run_query

# ── Page Config ──────────────────────────────
st.set_page_config(
    page_title="PhysicsBot — Study Buddy",
    page_icon="⚛️",
    layout="centered",
)

# ── Custom CSS ────────────────────────────────
st.markdown("""
<style>
.main { background-color: #f5f7ff; }
.title-block {
    background: linear-gradient(135deg, #4f46e5, #7c3aed);
    padding: 22px 30px; border-radius: 14px;
    margin-bottom: 20px; color: white;
}
.meta-pill {
    display: inline-block; background: #ede9fe; color: #5b21b6;
    border-radius: 20px; padding: 2px 10px; font-size: 12px; margin-right: 6px;
}
.good-pill {
    display: inline-block; background: #d1fae5; color: #065f46;
    border-radius: 20px; padding: 2px 10px; font-size: 12px; margin-right: 6px;
}
.warn-pill {
    display: inline-block; background: #fef3c7; color: #92400e;
    border-radius: 20px; padding: 2px 10px; font-size: 12px; margin-right: 6px;
}
</style>
""", unsafe_allow_html=True)

# ── Header ────────────────────────────────────
st.markdown("""
<div class="title-block">
    <h2 style="margin:0">⚛️ PhysicsBot — Study Buddy</h2>
    <p style="margin:4px 0 0 0; opacity:0.85; font-size:14px">
        B.Tech Physics · RAG · LangGraph · Calculator Tool · Memory
    </p>
</div>
""", unsafe_allow_html=True)

# ── Load Agent (cached) ───────────────────────
@st.cache_resource(show_spinner="📚 Loading physics knowledge base...")
def load_agent():
    return initialize()

app = load_agent()

# ── Session State ─────────────────────────────
if "chat_history" not in st.session_state:
    st.session_state.chat_history = []

if "agent_state" not in st.session_state:
    st.session_state.agent_state = {
        "question": "", "messages": [], "route": "", "retrieved": "",
        "sources": [], "tool_result": "", "answer": "", "faithfulness": 0.0,
        "eval_retries": 0, "user_name": "",
    }

# ── Sidebar ───────────────────────────────────
with st.sidebar:
    st.markdown("### ⚛️ Topics Covered")
    topics = [
        "📐 Kinematics", "📏 Equations of Motion", "🔵 Newton's Laws",
        "⚡ Work, Energy & Power", "🔄 Circular Motion", "🌍 Gravitation",
        "〰️ Simple Harmonic Motion", "🌊 Waves", "📏 Units & Dimensions",
    ]
    for t in topics:
        st.markdown(f"- {t}")

    st.markdown("---")
    st.markdown("### 💡 Try These Questions")
    samples = [
        "What is Newton's second law?",
        "Explain SHM with formula",
        "What is the work-energy theorem?",
        "Calculate: force on 5kg body with 3 m/s² acceleration",
        "What is escape velocity?",
        "Explain waves and interference",
        "What are the equations of motion?",
        "What is today's date?",
    ]
    for q in samples:
        if st.button(q, key=f"s_{q}"):
            st.session_state["prefill"] = q

    st.markdown("---")
    col1, col2 = st.columns(2)
    with col1:
        if st.button("🗑️ New Chat"):
            import uuid
            st.session_state.thread_id = str(uuid.uuid4())
            st.session_state.chat_history = []
            st.session_state.agent_state = {
                "question": "", "messages": [], "route": "", "retrieved": "",
                "sources": [], "tool_result": "", "answer": "", "faithfulness": 0.0,
                "eval_retries": 0, "user_name": "",
            }
            st.rerun()

    if st.session_state.agent_state.get("user_name"):
        st.markdown(f"### 🎓 Student: **{st.session_state.agent_state['user_name']}**")

# ── Chat Display ──────────────────────────────
for role, text, meta in st.session_state.chat_history:
    avatar = "🎓" if role == "user" else "⚛️"
    with st.chat_message(role, avatar=avatar):
        st.write(text)
        if meta and role == "assistant":
            route = meta.get("route", "")
            faith = meta.get("faithfulness")
            sources = meta.get("sources", [])
            pills = ""
            if route:
                pills += f'<span class="meta-pill">🔀 {route}</span>'
            if faith is not None and route == "retrieve":
                cls = "good-pill" if faith >= 0.7 else "warn-pill"
                pills += f'<span class="{cls}">✅ Faithfulness: {faith:.0%}</span>'
            for s in sources:
                pills += f'<span class="meta-pill">📄 {s}</span>'
            if pills:
                st.markdown(pills, unsafe_allow_html=True)

# ── Chat Input ────────────────────────────────
prefill = st.session_state.pop("prefill", "")
user_input = st.chat_input("Ask about Newton's Laws, SHM, Waves, or solve a numerical...") or prefill

if user_input:
    st.session_state.chat_history.append(("user", user_input, {}))
    with st.chat_message("user", avatar="🎓"):
        st.write(user_input)

    with st.chat_message("assistant", avatar="⚛️"):
        with st.spinner("Thinking..."):
            result = run_query(app, user_input, st.session_state.agent_state)
            st.session_state.agent_state = result

        answer = result.get("answer", "Sorry, something went wrong. Please try again.")
        route = result.get("route", "")
        faith = result.get("faithfulness")
        sources = result.get("sources", [])

        st.write(answer)
        pills = ""
        if route:
            pills += f'<span class="meta-pill">🔀 {route}</span>'
        if faith is not None and route == "retrieve":
            cls = "good-pill" if faith >= 0.7 else "warn-pill"
            pills += f'<span class="{cls}">✅ Faithfulness: {faith:.0%}</span>'
        for s in sources:
            pills += f'<span class="meta-pill">📄 {s}</span>'
        if pills:
            st.markdown(pills, unsafe_allow_html=True)

    st.session_state.chat_history.append(
        ("assistant", answer, {"route": route, "faithfulness": faith, "sources": sources})
    )

# ── Footer ────────────────────────────────────
st.markdown("---")
st.markdown(
    "<p style='text-align:center;color:#888;font-size:12px;'>"
    "⚛️ PhysicsBot covers B.Tech 1st/2nd year physics only. "
    "Always verify numerical answers with your textbook."
    "</p>", unsafe_allow_html=True
)
