import streamlit as st
import requests
from pathlib import Path

API_BASE = "http://localhost:8000"

st.set_page_config(page_title="RAG Assistant", page_icon="🔍", layout="wide")
st.title("🔍 Conversational RAG Assistant")


# ── Session State Init ─────────────────────────────────────────────────────────
if "session_id" not in st.session_state:
    st.session_state.session_id = None
if "user_id" not in st.session_state:
    st.session_state.user_id = None
if "messages" not in st.session_state:
    st.session_state.messages = []


# ── Sidebar ────────────────────────────────────────────────────────────────────
with st.sidebar:
    st.header("⚙️ Setup")

    user_id = st.text_input("User ID", value=st.session_state.user_id or "", placeholder="e.g. user_123")

    if st.button("Start New Session", type="primary") and user_id:
        resp = requests.post(f"{API_BASE}/sessions", json={"user_id": user_id})
        if resp.ok:
            data = resp.json()
            st.session_state.session_id = data["session_id"]
            st.session_state.user_id = user_id
            st.session_state.messages = []
            st.success(f"Session started!")
            st.code(data["session_id"], language=None)
        else:
            st.error("Failed to create session")

    if st.session_state.session_id:
        st.divider()
        st.caption(f"**Session:** `{st.session_state.session_id[:8]}...`")
        st.caption(f"**User:** `{st.session_state.user_id}`")

    st.divider()
    st.header("📄 Upload Document")
    uploaded = st.file_uploader(
        "PDF, Markdown, HTML, or TXT",
        type=["pdf", "md", "html", "htm", "txt"],
    )
    if st.button("Process Document") and uploaded:
        with st.spinner("Parsing and indexing..."):
            resp = requests.post(
                f"{API_BASE}/documents/upload",
                files={"file": (uploaded.name, uploaded.getvalue(), uploaded.type)},
            )
        if resp.ok:
            d = resp.json()
            st.success(f"✅ Indexed {d['chunks_indexed']} chunks from `{d['filename']}`")
        else:
            st.error(f"Upload failed: {resp.text}")

    # Load previous sessions
    if user_id and st.button("Load My Sessions"):
        resp = requests.get(f"{API_BASE}/users/{user_id}/sessions")
        if resp.ok and resp.json():
            sessions = resp.json()
            st.subheader("Previous Sessions")
            for s in sessions[:5]:
                label = f"{s['session_id'][:8]}... ({s['updated_at'][:10]})"
                if st.button(label, key=s["session_id"]):
                    st.session_state.session_id = s["session_id"]
                    st.session_state.user_id = user_id
                    # Load history
                    hist_resp = requests.get(f"{API_BASE}/sessions/{s['session_id']}/history")
                    if hist_resp.ok:
                        msgs = hist_resp.json()["messages"]
                        st.session_state.messages = [
                            {"role": m["role"], "content": m["content"]} for m in msgs
                        ]
                    st.rerun()


# ── Chat Interface ─────────────────────────────────────────────────────────────
if not st.session_state.session_id:
    st.info("👈 Enter your User ID and click **Start New Session** to begin.")
else:
    # Render message history
    for msg in st.session_state.messages:
        with st.chat_message(msg["role"]):
            st.markdown(msg["content"])

    # Chat input
    if prompt := st.chat_input("Ask something about your documents..."):
        # Show user message immediately
        st.session_state.messages.append({"role": "user", "content": prompt})
        with st.chat_message("user"):
            st.markdown(prompt)

        # Call API
        with st.chat_message("assistant"):
            with st.spinner("Thinking..."):
                resp = requests.post(
                    f"{API_BASE}/chat",
                    json={
                        "session_id": st.session_state.session_id,
                        "user_id": st.session_state.user_id,
                        "query": prompt,
                    },
                )

            if resp.ok:
                data = resp.json()
                answer = data["response"]
                st.markdown(answer)

                # Show sources in expander
                if data.get("sources"):
                    with st.expander(f"📚 Sources ({len(data['sources'])})", expanded=False):
                        for s in data["sources"]:
                            st.markdown(
                                f"**[{s['index']}]** `{s['source']}` — "
                                f"*{s.get('section', '')}* (score: {s['score']})"
                            )

                # Show debug info
                if data.get("rewritten_query") and data["rewritten_query"] != prompt:
                    with st.expander("🔄 Query rewritten", expanded=False):
                        st.caption(data["rewritten_query"])

                st.session_state.messages.append({"role": "assistant", "content": answer})
            else:
                err = f"Error: {resp.status_code} — {resp.text}"
                st.error(err)
                st.session_state.messages.append({"role": "assistant", "content": err})
