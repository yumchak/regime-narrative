# Deploying the dashboard

The report is already a shareable static page. **The dashboard is not** — it is a
Python server, so it needs a host that runs Python. This is the five-minute
version.

I cannot do this step for you: it requires signing in to your own GitHub and
Streamlit accounts. Everything the host needs is already committed.

---

## Streamlit Community Cloud — free, and the right fit

1. Go to **share.streamlit.io** and sign in with GitHub.
2. **Create app** → **Deploy a public app from GitHub**.
3. Fill in:
   - Repository: `yumchak/regime-narrative`
   - Branch: `main`
   - Main file path: `app.py`
4. **Deploy.** First build takes a few minutes while it installs
   `requirements.txt`.

You get a URL like `https://regime-narrative.streamlit.app`. Share that.

### Do not add your API key to Streamlit secrets

This is the one thing that can cost you real money.

A public Streamlit app is public. If you put `ANTHROPIC_API_KEY` in the app's
secrets, **every visitor spends from your account** — the Live view and Your data
both make model calls, and there is no rate limit or login in front of them.

The app is built so you do not have to. The sidebar has a key field that holds
the value in that visitor's browser session only. Each person brings their own
key; nobody can bill you. Leave secrets empty.

---

## What works on the deployed app, and what does not

Everything read-only works immediately, because the results are committed:

| View | Deployed behaviour |
|---|---|
| Overview | Full. Reads `outputs/regimes.json`. |
| Transitions | Full. All 20 transitions with their explanations. |
| Controls | Full. Every control result, both models. |
| Provenance | Full, except the call log, which lives in the local cache. |
| **Live** | Works with a visitor's own key. First run on any date takes ~15s while it fetches 14 Wikipedia pages — the cache is empty on a fresh host. |
| **Your data** | Same. Works, but every window is a cold fetch. |

Verified by cloning the repo to a clean directory with no `data/cache` and
rendering all six views.

### The cold-cache caveat

`data/cache/` is gitignored — it holds about 1,100 API responses and would bloat
the repository. So the deployed app starts with nothing cached and re-fetches on
demand. Read-only views are unaffected; only Live and Your data pay the wait.

If you want the demo instant on the deployed app, commit the cache for the two
dates you show on camera:

```bash
git add -f data/cache/news/wikipedia_raw
git commit -m "Ship the news cache so the hosted demo is instant"
```

That adds a few megabytes. Worth it if you are handing the link to judges;
skip it otherwise.

---

## Alternatives, if Streamlit Cloud does not suit

- **Hugging Face Spaces** — free, pick the Streamlit SDK, same repo layout works.
- **Render / Railway** — free tiers, run
  `streamlit run app.py --server.port $PORT --server.address 0.0.0.0`.
- **Local for the video** — honestly fine. Judges watch a recording; a local app
  records identically to a hosted one, and the report link covers sharing.

---

## Committed for the host

| File | Why |
|---|---|
| `requirements.txt` | Pinned runtime dependencies. A test asserts every imported package appears here — it was once pinned before `streamlit` and `plotly` were added, which would have failed the build. |
| `requirements-dev.txt` | Test-only extras, not installed on the host. |
| `runtime.txt` | `python-3.12`. 3.14 has no `hmmlearn` wheel. |
| `.streamlit/config.toml` | Theme, 5 MB upload cap, usage stats off. |
| `outputs/*.json`, `*.png`, `oos_states.csv` | The results the read-only views render. |
