import streamlit as st
import requests
import json
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

        # Call streaming API
        with st.chat_message("assistant"):
            payload = {
                "session_id": st.session_state.session_id,
                "user_id": st.session_state.user_id,
                "query": prompt,
            }

            done_meta: dict = {}       # filled by the generator's side-effect
            answer_parts: list[str] = []

            def token_generator():
                """Yield text tokens from /chat/stream; capture done metadata."""
                with requests.post(
                    f"{API_BASE}/chat/stream",
                    json=payload,
                    stream=True,
                    timeout=120,
                ) as r:
                    r.raise_for_status()
                    for raw_line in r.iter_lines():
                        if not raw_line:
                            continue
                        try:
                            chunk = json.loads(raw_line)
                        except json.JSONDecodeError:
                            continue

                        if chunk.get("type") == "token":
                            token = chunk["content"]
                            answer_parts.append(token)
                            yield token                  # Streamlit renders this live

                        elif chunk.get("type") == "done":
                            done_meta.update(chunk)      # capture sources / metadata
                            # cache hit: no tokens were streamed — yield full response
                            if chunk.get("from_cache") and chunk.get("response"):
                                answer_parts.append(chunk["response"])
                                yield chunk["response"]

            try:
                # st.write_stream() consumes the generator and renders tokens live
                st.write_stream(token_generator())
                answer = "".join(answer_parts)

                # ── Sources ───────────────────────────────────────────────────
                if done_meta.get("sources"):
                    with st.expander(
                        f"📚 Sources ({len(done_meta['sources'])})", expanded=False
                    ):
                        for s in done_meta["sources"]:
                            st.markdown(
                                f"**[{s['index']}]** `{s['source']}` — "
                                f"*{s.get('section', '')}* (score: {s['score']})"
                            )

                # ── Cache hit badge ───────────────────────────────────────────
                if done_meta.get("from_cache"):
                    st.caption(
                        f"⚡ Answered from semantic cache "
                        f"(score: {done_meta.get('cache_score', ''):.2f})"
                    )

                # ── Rewrite debug ─────────────────────────────────────────────
                rq = done_meta.get("rewritten_query")
                if rq and rq != prompt:
                    with st.expander("🔄 Query rewritten", expanded=False):
                        st.caption(rq)

                st.session_state.messages.append({"role": "assistant", "content": answer})

            except Exception as e:
                err = f"Stream error: {e}"
                st.error(err)
                st.session_state.messages.append({"role": "assistant", "content": err})

