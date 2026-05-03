import streamlit as st
import requests
import json

API_BASE = "http://localhost:8000"

st.set_page_config(page_title="RAG Assistant", page_icon="🔍", layout="wide")
st.title("🔍 Conversational RAG Assistant")

# ── Session State ─────────────────────────────────────────────────────────────
if "session_id" not in st.session_state:
    st.session_state.session_id = None
if "user_id" not in st.session_state:
    st.session_state.user_id = None
if "messages" not in st.session_state:
    st.session_state.messages = []

# ── Sidebar ───────────────────────────────────────────────────────────────────
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
            st.success("Session started!")
            st.code(data["session_id"], language=None)
        else:
            st.error("Failed to create session")

    if st.session_state.session_id:
        st.divider()
        st.caption(f"**Session:** `{st.session_state.session_id[:8]}...`")
        st.caption(f"**User:** `{st.session_state.user_id}`")

    st.divider()
    st.header("📄 Upload Document")
    uploaded = st.file_uploader("PDF, Markdown, HTML, or TXT", type=["pdf", "md", "html", "htm", "txt"])
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

    if user_id and st.button("Load My Sessions"):
        resp = requests.get(f"{API_BASE}/users/{user_id}/sessions")
        if resp.ok and resp.json():
            st.subheader("Previous Sessions")
            for s in resp.json()[:5]:
                label = f"{s['session_id'][:8]}... ({s['updated_at'][:10]})"
                if st.button(label, key=s["session_id"]):
                    st.session_state.session_id = s["session_id"]
                    st.session_state.user_id = user_id
                    hist_resp = requests.get(f"{API_BASE}/sessions/{s['session_id']}/history")
                    if hist_resp.ok:
                        msgs = hist_resp.json()["messages"]
                        st.session_state.messages = [
                            {"role": m["role"], "content": m["content"]} for m in msgs
                        ]
                    st.rerun()


# ── Chat ──────────────────────────────────────────────────────────────────────
if not st.session_state.session_id:
    st.info("👈 Enter your User ID and click **Start New Session** to begin.")
else:
    for msg in st.session_state.messages:
        with st.chat_message(msg["role"]):
            st.markdown(msg["content"])

    if prompt := st.chat_input("Ask something about your documents..."):
        st.session_state.messages.append({"role": "user", "content": prompt})
        with st.chat_message("user"):
            st.markdown(prompt)

        with st.chat_message("assistant"):
            token_placeholder = st.empty()
            sources_placeholder = st.empty()

            full_text = ""
            sources = []
            from_cache = False
            cache_score = None

            # ── SSE streaming request ─────────────────────────────────────────
            with requests.post(
                f"{API_BASE}/chat/stream",
                json={
                    "session_id": st.session_state.session_id,
                    "user_id": st.session_state.user_id,
                    "query": prompt,
                },
                stream=True,
                timeout=120,
            ) as resp:
                if resp.ok:
                    for raw_line in resp.iter_lines():
                        if not raw_line:
                            continue
                        line = raw_line.decode("utf-8") if isinstance(raw_line, bytes) else raw_line
                        if not line.startswith("data: "):
                            continue

                        chunk = json.loads(line[6:])

                        if chunk["type"] == "meta":
                            sources = chunk.get("sources", [])
                            from_cache = chunk.get("from_cache", False)
                            cache_score = chunk.get("cache_score")

                        elif chunk["type"] == "token":
                            full_text += chunk["content"]
                            token_placeholder.markdown(full_text + "▌")  # typing cursor

                        elif chunk["type"] == "done":
                            token_placeholder.markdown(full_text)  # final render, remove cursor

                    # ── Sources ───────────────────────────────────────────────
                    with sources_placeholder.container():
                        if sources:
                            label = f"📚 Sources ({len(sources)})"
                            if from_cache:
                                label += f"  ⚡ cached (similarity {cache_score})"
                            with st.expander(label, expanded=False):
                                for s in sources:
                                    st.markdown(
                                        f"**[{s['index']}]** `{s['source']}` — "
                                        f"*{s.get('section', '')}* (score: {s['score']})"
                                    )
                else:
                    full_text = f"Error {resp.status_code}: {resp.text}"
                    token_placeholder.error(full_text)

        st.session_state.messages.append({"role": "assistant", "content": full_text})