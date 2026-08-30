"""regime-narrative — interactive dashboard.

    streamlit run app.py

Six views, in the order someone should meet the project:

    Overview     what the model found, and how strong the separation is
    Transitions  one detected change at a time: the news it saw, the
                 explanation it produced, and whether each claim is grounded
    Controls     the three controls and the blind-matching test
    Your data    the same pipeline pointed at anyone else's regime dates
    Live         pick any date, watch the pipeline run end to end
    Provenance   the prompt, the settings, the call log

The Live view is the one that shows the architecture rather than describing it:
the HMM has already decided which days are transitions, and the language model
only ever sees a boundary-enforced window of news. You can point it at an
ordinary Tuesday and watch it decline.

Everything except Live and Your data reads from cached JSON and needs no API key.
The key, if supplied, is held in session state only -- never written to disk.
"""

from __future__ import annotations

import io
import json
import os
import sys
import warnings
from datetime import date, timedelta
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent / "src"))
warnings.filterwarnings("ignore")

import pandas as pd
import plotly.graph_objects as go
import streamlit as st

from regime_narrative.config import has_env, load_settings, output_dir

st.set_page_config(
    page_title="regime-narrative",
    page_icon="◪",
    layout="wide",
    initial_sidebar_state="expanded",
)

CALM = "#dfe9f2"
STRESSED = "#f6cfd0"
INK = "#1d2b36"
RED = "#b4343a"
BLUE = "#4a6fa5"

st.markdown(
    """
    <style>
      .block-container {padding-top: 2.2rem; max-width: 1400px;}
      h1 {font-size: 1.9rem !important; letter-spacing: -0.02em;}
      h2 {font-size: 1.25rem !important; margin-top: 1.8rem;}
      h3 {font-size: 1.02rem !important;}
      [data-testid="stMetricValue"] {font-size: 1.65rem;}
      [data-testid="stMetricLabel"] {font-size: 0.74rem; text-transform: uppercase;
        letter-spacing: 0.05em; opacity: 0.72;}
      .caveat {border-left: 3px solid #a8641c; background: rgba(168,100,28,0.07);
        padding: 0.75rem 1rem; border-radius: 0 4px 4px 0; font-size: 0.9rem;}
      .good {border-left: 3px solid #2e7d55; background: rgba(46,125,85,0.07);
        padding: 0.75rem 1rem; border-radius: 0 4px 4px 0; font-size: 0.9rem;}
      .newsitem {border-left: 2px solid #c8d2db; padding: 0.35rem 0 0.35rem 0.8rem;
        margin: 0.35rem 0; font-size: 0.87rem;}
      .cited {border-left-color: #b4343a; background: rgba(180,52,58,0.05);}
      .idtag {font-family: ui-monospace, monospace; font-size: 0.72rem; opacity: 0.55;}
    </style>
    """,
    unsafe_allow_html=True,
)


# ---------------------------------------------------------------------------
# data loading
# ---------------------------------------------------------------------------


@st.cache_data(show_spinner=False)
def load_json(name: str):
    p = output_dir() / name
    if not p.exists():
        return None
    return json.loads(p.read_text(encoding="utf-8"))


@st.cache_data(show_spinner=False)
def load_states() -> pd.DataFrame | None:
    p = output_dir() / "oos_states.csv"
    if not p.exists():
        return None
    return pd.read_csv(p, index_col=0, parse_dates=True)


REGIMES = load_json("regimes.json")
NEWS = load_json("news_manifest.json")
CEILING = load_json("blind_match_ceiling.json")
NARR = load_json("narrative_results.json")
RETEST = load_json("retest_stability.json")
POWER = load_json("placebo_power.json")
REFEREE = load_json("referee_stats.json")
STATES = load_states()
CFG = load_settings()

if REGIMES is None:
    st.error("outputs/regimes.json not found — run `python scripts/01_regimes.py` first.")
    st.stop()

NARRATIVES = (NARR or {}).get("narratives", {})
TRANSITIONS = REGIMES["seed_stability"]["transitions"]


def fmt(v, nd=2, suffix=""):
    if v is None or (isinstance(v, float) and v != v):
        return "—"
    if isinstance(v, float):
        return f"{v:.{nd}f}{suffix}"
    return f"{v}{suffix}"


# ---------------------------------------------------------------------------
# sidebar
# ---------------------------------------------------------------------------

with st.sidebar:
    st.markdown("### regime-narrative")
    st.caption(
        "A statistical model decides **when** the regime changed. "
        "A language model reading only prior news describes **what was happening**. "
        "Every claim about the second half is measured."
    )
    view = st.radio(
        "View",
        ["Overview", "Transitions", "Controls", "Your data", "Live", "Provenance"],
        label_visibility="collapsed",
    )
    st.divider()

    # --- API key -----------------------------------------------------------
    # Held in session state for this browser session only. Never written to
    # disk, never logged, and not persisted across a restart -- a key in a repo
    # is a key in someone's git history.
    if not has_env("ANTHROPIC_API_KEY"):
        with st.expander("🔑 Anthropic API key", expanded=False):
            entered = st.text_input(
                "Key", type="password", label_visibility="collapsed",
                placeholder="sk-ant-...",
                help="Session-only. Not written to disk and not logged. "
                     "Needed for the Live view and for explaining your own data.",
            )
            if entered:
                os.environ["ANTHROPIC_API_KEY"] = entered.strip()
                st.success("Key set for this session")
    key_ok = has_env("ANTHROPIC_API_KEY")
    st.caption(f"API key: {'**set**' if key_ok else '**not set** — generation disabled'}")

    st.divider()
    st.caption(
        f"Study: **{REGIMES['data']['n_days']:,}** trading days · "
        f"**{REGIMES['walk_forward']['n_folds']}** folds · "
        f"**{len(TRANSITIONS)}** transitions"
    )
    if NARR:
        st.caption(f"Narratives: **{len(NARRATIVES)}** · `{NARR.get('model')}`")
    else:
        st.caption("Narratives: **not generated**")


# ---------------------------------------------------------------------------
# shared chart
# ---------------------------------------------------------------------------


def regime_figure(highlight: str | None = None, height: int = 430) -> go.Figure:
    df = STATES
    fig = go.Figure()

    values = df["state"].values.astype(int)
    idx = df.index
    start = 0
    for i in range(1, len(values) + 1):
        if i == len(values) or values[i] != values[start]:
            fig.add_vrect(
                x0=idx[start], x1=idx[min(i, len(idx) - 1)],
                fillcolor=STRESSED if values[start] == 1 else CALM,
                opacity=0.85, line_width=0, layer="below",
            )
            start = i

    fig.add_trace(go.Scatter(
        x=idx, y=df["spy"], mode="lines",
        line=dict(color=INK, width=1.1), name="SPY",
        hovertemplate="%{x|%Y-%m-%d}<br>SPY %{y:.2f}<extra></extra>",
    ))

    for t in TRANSITIONS:
        d = pd.Timestamp(t["date"])
        if not (idx[0] <= d <= idx[-1]):
            continue
        is_hi = highlight == t["date"]
        to_stress = t["direction"].startswith("calm")
        fig.add_vline(
            x=d,
            line=dict(
                color="#111" if is_hi else (RED if to_stress else BLUE),
                width=2.6 if is_hi else 0.9,
                dash=None if is_hi else "solid",
            ),
        )

    fig.update_layout(
        height=height, margin=dict(l=8, r=8, t=10, b=8),
        yaxis=dict(type="log", title="SPY (log)"),
        xaxis=dict(title=None),
        showlegend=False, plot_bgcolor="white", hovermode="x unified",
    )
    return fig


# ---------------------------------------------------------------------------
# Overview
# ---------------------------------------------------------------------------

if view == "Overview":
    st.title("Regime detection with a measured explanation layer")

    oos, ins = REGIMES["out_of_sample"], REGIMES["in_sample"]
    fwd, wf = REGIMES["out_of_sample_forward"], REGIMES["walk_forward"]
    pf = REGIMES["persistence_filter"]

    c = st.columns(5)
    c[0].metric("Out-of-sample vol ratio", f"{oos['ratio']:.2f}×",
                help=f"95% CI [{oos['ratio_ci_low']:.2f}, {oos['ratio_ci_high']:.2f}]. "
                     f"Block bootstrap, because daily returns are serially dependent.")
    c[1].metric("Forward-20d ratio", f"{fwd['ratio']:.2f}×",
                help="Volatility that had not happened yet when the state was "
                     "assigned. The non-circular version of the claim.")
    c[2].metric("Per-fold median", f"{wf['per_fold_ratio_median']:.2f}×",
                help="Within-fold separation. Lower than the pooled figure because "
                     "pooling mixes calm-era and crisis-era days.")
    c[3].metric("In-sample (prior work)", f"{ins['ratio']:.2f}×",
                delta=f"{oos['ratio'] - ins['ratio']:+.2f} vs out-of-sample",
                delta_color="off")
    c[4].metric("Transitions", len(TRANSITIONS),
                help=f"{pf['raw_state_flips']} raw flips → "
                     f"{pf['flips_after_filter']} after persistence filtering → "
                     f"{len(TRANSITIONS)} stable across seeds.")

    st.markdown(
        '<div class="caveat"><b>The prior notebook reported 2.33× from an HMM fitted '
        "once over the full history, and separately ran 26 walk-forward folds for a "
        "return classifier. Those were two different experiments.</b> Here the HMM is "
        "refitted inside every fold, the scaler is fitted per fold, and states are "
        "assigned by a forward filter that cannot see the future — so the transition "
        "dates were knowable on the day.</div>",
        unsafe_allow_html=True,
    )

    st.plotly_chart(regime_figure(), width='stretch')

    left, right = st.columns([1.15, 1])
    with left:
        st.subheader("Markets it was never fitted to")
        rows = [{
            "Index": "SPY (fitted)", "Folds": wf["n_folds"],
            "Same-day": oos["ratio"], "Forward 20d": fwd["ratio"],
            "Per-fold median": wf["per_fold_ratio_median"],
        }]
        for g in REGIMES["generalisation"]:
            if "error" in g:
                continue
            rows.append({
                "Index": g["index"], "Folds": g["n_folds"],
                "Same-day": g["ratio_oos"], "Forward 20d": g["ratio_forward"],
                "Per-fold median": g["per_fold_median"],
            })
        st.dataframe(
            pd.DataFrame(rows).style.format({
                "Same-day": "{:.2f}×", "Forward 20d": "{:.2f}×",
                "Per-fold median": "{:.2f}×"}),
            hide_index=True, width='stretch',
        )
        st.caption(
            "The same procedure refitted without modification. Separation holds on "
            "all three, but weaker than on the fitted universe — stated because it "
            "is more credible than three tidy successes."
        )

    with right:
        st.subheader("Persistence filtering")
        sens = pd.DataFrame(REGIMES["dwell_sensitivity"])
        fig = go.Figure()
        fig.add_trace(go.Scatter(
            x=sens["min_dwell_days"], y=sens["n_transitions"],
            mode="lines+markers", line=dict(color=INK, width=2),
            marker=dict(size=7),
        ))
        chosen = CFG["transitions"]["min_dwell_days"]
        fig.add_vline(x=chosen, line=dict(color=RED, width=2, dash="dash"),
                      annotation_text=f"pre-registered: {chosen}d")
        fig.update_layout(
            height=260, margin=dict(l=8, r=8, t=28, b=8),
            xaxis_title="Minimum dwell (trading days)",
            yaxis_title="Transitions", plot_bgcolor="white", showlegend=False,
        )
        st.plotly_chart(fig, width='stretch')
        st.caption(
            f"A single-day flip is noise. {pf['raw_state_flips']} raw flips reduce to "
            f"{pf['flips_after_filter']} at the pre-registered threshold. The count is "
            "shown across thresholds so the choice is visible rather than convenient."
        )


# ---------------------------------------------------------------------------
# Transitions
# ---------------------------------------------------------------------------

elif view == "Transitions":
    st.title("Transitions")
    st.caption(
        "One detected regime change at a time: the news the model was given, the "
        "explanation it produced, and whether each claim is supported by what it cited."
    )

    labels = [
        f"{t['date']}  ·  {'into stress' if t['direction'].startswith('calm') else 'out of stress'}"
        for t in TRANSITIONS
    ]
    pick = st.selectbox("Transition", range(len(TRANSITIONS)),
                        format_func=lambda i: labels[i])
    t = TRANSITIONS[pick]
    wid = f"transition_{t['date']}"
    nar = NARRATIVES.get(wid)

    st.plotly_chart(regime_figure(highlight=t["date"], height=300),
                    width='stretch')

    m = st.columns(4)
    m[0].metric("Date", t["date"])
    m[1].metric("Direction", "into stress" if t["direction"].startswith("calm") else "out of stress")
    m[2].metric("Seed support", f"{int(round(100 * (t.get('seed_support') or 0)))}%",
                help="Fraction of random seeds that produced this transition.")
    m[3].metric("Dwell after", f"{t['dwell_after']}d")

    wmeta = (NEWS or {}).get("windows", {}).get(wid, {})
    if wmeta:
        st.caption(
            f"News window **{wmeta['window_start_utc'][:10]} → "
            f"{wmeta['window_end_utc'][:10]}** · {wmeta['n_items']} items · "
            f"{wmeta['n_generation']} shown to the model, {wmeta['n_holdout']} held out "
            f"for blind matching · {wmeta['dropped_post_boundary']} items post-dated the "
            f"boundary and were dropped"
        )

    if not nar:
        st.info("No narrative generated for this transition yet.")
    else:
        st.subheader("Generated explanation")
        if not nar["driver_identified"]:
            st.warning("**No identifiable driver.** The model declined.")
            if nar.get("insufficient_evidence_reason"):
                st.write(nar["insufficient_evidence_reason"])
        else:
            badge = {"high": "🟢 high", "medium": "🟡 medium",
                     "low": "🟠 low", "none": "⚪ none"}.get(nar["confidence"], nar["confidence"])
            st.markdown(f"**Confidence: {badge}**")
            st.write(nar["summary"])
            if nar.get("mechanism"):
                st.markdown(f"**Transmission mechanism.** {nar['mechanism']}")
            if nar.get("competing_explanation"):
                with st.expander("Strongest competing explanation"):
                    st.write(nar["competing_explanation"])

        if nar.get("claims"):
            st.subheader("Claims and their citations")
            fa = None
            if NARR:
                for r in NARR.get("faithfulness", {}).get("per_window", []):
                    if r["window_id"] == wid:
                        fa = r
                        break
            audits = {a["claim"]: a for a in (fa or {}).get("claims", [])}
            for cl in nar["claims"]:
                a = audits.get(cl["claim"])
                if a:
                    icon = {"grounded": "✅", "ungrounded": "⚠️",
                            "uncited": "❌", "fabricated_citation": "🚫"}.get(a["status"], "•")
                    tail = f"  ·  overlap {a['overlap']:.0%}  ·  {a['status']}"
                else:
                    icon, tail = "•", ""
                st.markdown(
                    f"{icon} {cl['claim']}<br>"
                    f"<span class='idtag'>cites {', '.join(cl['item_ids']) or 'nothing'}{tail}</span>",
                    unsafe_allow_html=True,
                )
            if fa:
                st.caption(
                    f"Grounding rate for this window: **{fa['grounding_rate']:.0%}** "
                    f"({fa['n_grounded']}/{fa['n_claims']} claims). Grounding is scored "
                    "by lexical overlap with the cited item, so the check itself has no "
                    "world knowledge and cannot be fooled the way an LLM judge could."
                )


# ---------------------------------------------------------------------------
# Controls
# ---------------------------------------------------------------------------

elif view == "Controls":
    st.title("Controls")
    st.caption(
        "These matter more than the narratives. A narrative engine that explains "
        "everything explains nothing — these measure which one this is."
    )

    tab1, tab2, tab3, tab4 = st.tabs(
        ["Hindsight", "Placebo", "Blind matching", "Memorisation"]
    )

    with tab1:
        nsum = (NEWS or {}).get("summary", {})
        leak = (NEWS or {}).get("hindsight_leak", {})
        st.subheader("The retrieval boundary")
        c = st.columns(4)
        c[0].metric("Windows", nsum.get("n_windows", "—"))
        c[1].metric("Retrieval failures", nsum.get("n_failures", "—"))
        c[2].metric("Total items", f"{nsum.get('total_items', 0):,}")
        c[3].metric("Post-boundary survivors", "0" if nsum.get("any_post_boundary_items") is False else "!")
        st.markdown(
            '<div class="good">The boundary is enforced structurally, not by '
            "convention: a window object refuses to construct if any item post-dates "
            "it. 26 tests enforce this, including timezone edge cases.</div>",
            unsafe_allow_html=True,
        )
        st.subheader("Why revision pinning is not optional")
        c = st.columns(2)
        c[0].metric("Mean of page written AFTER the boundary",
                    f"{leak.get('mean_pct_written_after_boundary', '—')}%")
        c[1].metric("Worst case", f"{leak.get('max_pct_written_after_boundary', '—')}%")
        st.caption(
            "Wikipedia's page for a given day keeps being edited for years. Fetching "
            "the live page would feed the model text written with full knowledge of "
            "what followed, while the manifest recorded the window as closing on the "
            "transition date. Every page here is pinned to its revision id."
        )
        samples = [s for s in leak.get("samples", []) if s.get("status") == "ok"]
        if samples:
            st.dataframe(pd.DataFrame([{
                "Page": s["page"].split("/")[-1],
                "Pinned (bytes)": s["pinned_size_bytes"],
                "Today (bytes)": s["current_size_bytes"],
                "Last edited": s["current_timestamp"][:10],
                "Written after": f"{s['pct_written_after_boundary']}%",
            } for s in samples]), hide_index=True, width='stretch')

    with tab2:
        bal = REGIMES["placebo"]["balance"]
        st.subheader("Era-matched control dates")
        c = st.columns(4)
        c[0].metric("Controls", bal["n_placebos"])
        c[1].metric("Era balance (year gap)", fmt(bal["year_mean_gap"]))
        c[2].metric("Mean offset", f"{bal['mean_abs_offset_days']}d")
        c[3].metric("Clean / contaminated",
                    f"{bal['stratum_split']['clean']} / {bal['stratum_split']['contaminated']}")
        st.caption(
            "Sampling controls from quiet periods would bias the test toward passing: "
            "transitions cluster in crisis years, quiet periods do not, and you would "
            "be measuring the era rather than the regime. Each control is drawn from "
            "its partner transition's own neighbourhood, and controls are disjoint "
            "from each other as well as from transitions."
        )
        if NARR:
            p = NARR["placebo"]
            rows = [
                ("Real transitions", p["transitions"]),
                ("All controls", p["placebos_all"]),
                ("Clean controls (primary)", p["placebos_clean_stratum"]),
            ]
            st.dataframe(pd.DataFrame([{
                "Arm": name, "n": d.get("n"),
                "Confident": d.get("n_confident"),
                "Rate": d.get("confident_rate"),
                "Driver identified": d.get("driver_rate"),
            } for name, d in rows]), hide_index=True, width='stretch')
            c = st.columns(2)
            c[0].metric("Fisher p — vs all controls", fmt(p.get("fisher_p_all"), 4))
            c[1].metric("Fisher p — vs clean controls", fmt(p.get("fisher_p_clean"), 4))
            st.caption(
                "Identical prompt and identical pipeline in both arms; only the news "
                "differs. This single comparison is the most important number here."
            )

            # A null result means nothing without the power behind it. The comparison
            # above did not separate; the only honest way to report that is next to
            # what this design could ever have detected -- which is almost nothing.
            if POWER:
                pw = POWER.get("power_at_observed_effect", {}).get("vs_clean_n13")
                obs = POWER.get("observed", {})
                st.markdown("**What could this test ever have detected?**")
                q = st.columns(3)
                q[0].metric("Observed gap", f"{obs.get('gap_pp')} pp",
                            delta=f"{100 * obs.get('transition_rate', 0):.0f}% vs "
                                  f"{100 * obs.get('clean_control_rate', 0):.0f}%",
                            delta_color="off")
                q[1].metric("Power at that gap", f"{pw:.2f}",
                            delta=f"about 1 in {round(1 / pw)}" if pw else None,
                            delta_color="off")
                q[2].metric("Binding constraint",
                            f"{NARR['placebo']['transitions']['n']} transitions",
                            delta="not the controls", delta_color="off")
                st.markdown(
                    f'<div class="caveat"><b>Underpowered by design, not by accident.</b> '
                    f'If the {obs.get("gap_pp")}-point gap above is real, this study had '
                    f'roughly a <b>1 in {round(1 / pw)}</b> chance of detecting it. The '
                    f"correct reading is <i>difference not resolved</i> &mdash; not "
                    f"<i>no difference exists</i>. More control dates cannot fix this; "
                    f"only more transitions can.</div>",
                    unsafe_allow_html=True,
                )
                curve = POWER.get("power_curve_vs_clean_controls", {})
                if curve:
                    st.caption("Power against a hypothetical transition rate "
                               "(the non-circular version of the same question):")
                    st.dataframe(pd.DataFrame(
                        [{"If the transition rate were": k, "Power": v}
                         for k, v in curve.items()]),
                        hide_index=True, width='stretch')
        else:
            st.info("Awaiting narrative generation.")

    with tab3:
        st.subheader("Are the explanations specific, or macro boilerplate?")
        st.caption(
            "Explanations are stripped of dates, scrambled, and matched back to their "
            "window using BM25 against **held-out** articles the model never saw. The "
            "matcher has no world knowledge, so it cannot date an explanation from "
            "training the way an LLM judge could."
        )
        if CEILING:
            arms = CEILING.get("section_arms", {})
            st.markdown("**Ceiling — the best any explanation could do**")
            st.dataframe(pd.DataFrame([{
                "Sections": lbl,
                "n": arms.get(k, {}).get("n_scored"),
                "Accuracy": arms.get(k, {}).get("accuracy"),
                "Chance": arms.get(k, {}).get("chance"),
                "p": arms.get(k, {}).get("p_value"),
                "Median holdout words": arms.get(k, {}).get("median_holdout_words"),
            } for k, lbl in [("all_sections", "All"),
                             ("business_only", "Business only"),
                             ("non_business", "Non-business")] if k in arms]),
                hide_index=True, width='stretch')
            st.markdown(
                '<div class="caveat"><b>The most important caveat in the project.</b> '
                "The high ceiling is carried by generic world news, not market content. "
                "Business-only text matches near chance. Part of that is power — "
                "business holdouts have a median of 183 words against 2,772 — but "
                "either way, an explanation that obeys the prompt and discusses market "
                "drivers has comparatively little matchable text. A null result here "
                "would be ambiguous between &lsquo;the explanations are generic&rsquo; "
                "and &lsquo;the matchable signal is not where the explanation "
                "looks&rsquo;.</div>",
                unsafe_allow_html=True,
            )
            sweep = CEILING.get("length_sweep", [])
            if sweep:
                sdf = pd.DataFrame(sweep)
                fig = go.Figure()
                xs = [str(r["proxy_words"]) for r in sweep]
                fig.add_trace(go.Bar(x=xs, y=sdf["accuracy"], marker_color=BLUE,
                                     name="accuracy"))
                fig.add_hline(y=1 / max(1, len(TRANSITIONS)), line=dict(color=RED, dash="dash"),
                              annotation_text="chance")
                fig.update_layout(height=270, margin=dict(l=8, r=8, t=28, b=8),
                                  xaxis_title="Proxy explanation length (words)",
                                  yaxis_title="Match accuracy", plot_bgcolor="white",
                                  showlegend=False)
                st.plotly_chart(fig, width='stretch')
                st.caption(
                    "A real explanation is 100–200 words, where the ceiling is 40–70% "
                    "rather than 95%. Significance survives even at 60 words, so the "
                    "test stays well powered — the expectation just has to be right."
                )
        if NARR:
            bm = NARR.get("blind_match", {})
            st.markdown("**Measured on the real explanations**")
            st.dataframe(pd.DataFrame([{
                "Arm": lbl,
                "Correct": f"{bm.get(k, {}).get('n_correct')}/{bm.get(k, {}).get('n_scored')}",
                "Accuracy": bm.get(k, {}).get("accuracy"),
                "Chance": bm.get(k, {}).get("chance"),
                "MRR": bm.get(k, {}).get("mrr"),
                "p": bm.get(k, {}).get("p_value"),
            } for k, lbl in [
                ("transitions_all_sections", "Transitions, all sections"),
                ("transitions_business_only", "Transitions, business only"),
                ("transitions_non_business", "Transitions, non-business"),
                ("all_windows", "All windows pooled"),
            ] if isinstance(bm.get(k), dict) and "error" not in bm.get(k, {})]),
                hide_index=True, width='stretch')

        # 55%-against-5% is open to one objection: neighbouring windows share
        # running stories, so the matcher may only be recovering roughly WHEN.
        # Making each explanation compete against its nearest neighbours removes
        # that reading, and is the number worth quoting.
        if REFEREE and REFEREE.get("blind_match_hard_negatives"):
            hn = REFEREE["blind_match_hard_negatives"]
            st.markdown("**Against the hardest decoys — its own nearest neighbours**")
            st.dataframe(pd.DataFrame([{
                "Competing against": lbl,
                "Correct": f"{hn[k]['n_correct']}/{hn[k]['n_scored']}",
                "Accuracy": hn[k]["accuracy"],
                "Chance": hn[k]["chance"],
                "p": round(hn[k]["binomial_p"], 5),
            } for k, lbl in [("3_temporally_nearest", "3 nearest windows"),
                             ("5_temporally_nearest", "5 nearest windows")]
              if isinstance(hn.get(k), dict)]),
                hide_index=True, width='stretch')
            st.caption(hn.get("why", ""))

    with tab4:
        st.subheader("Memorisation")
        st.markdown(
            "The model has read text covering these events. Three mitigations, and an "
            "honest limit:\n\n"
            "- Absolute dates stripped from item text before the model sees it\n"
            "- Every claim must cite a supplied item, so unsupported recall is visible\n"
            "- Citation faithfulness measured lexically, with no world knowledge"
        )
        st.markdown(
            '<div class="caveat"><b>Not solved — mitigated and measured.</b> The '
            "obvious control, splitting results either side of the training cutoff, "
            "cannot work here: the post-cutoff arm holds a handful of transitions, and "
            "reporting <i>n</i>=3 as a test would be worse than not running it. "
            "Citation faithfulness replaces it because it scales to every claim.</div>",
            unsafe_allow_html=True,
        )
        if NARR:
            f = NARR["faithfulness"]
            st.dataframe(pd.DataFrame([{
                "Set": lbl, "Claims": f.get(k, {}).get("n_claims"),
                "Grounded": f.get(k, {}).get("grounding_rate"),
                "Ungrounded": f.get(k, {}).get("ungrounded_rate"),
                "Uncited": f.get(k, {}).get("uncited_rate"),
                "Fabricated citation": f.get(k, {}).get("fabricated_citation_rate"),
            } for k, lbl in [("transitions", "Transitions"),
                             ("placebos", "Controls"), ("all", "All")]]),
                hide_index=True, width='stretch')
            sens = f.get("threshold_sensitivity")
            if sens:
                st.caption("Grounding rate across overlap thresholds, so the headline "
                           "number cannot be read as a conveniently chosen cut-off.")
                st.dataframe(pd.DataFrame(sens), hide_index=True,
                             width='stretch')

        # A grounding rate means nothing without the rate everything scores.
        if REFEREE and REFEREE.get("grounding_null"):
            gn = REFEREE["grounding_null"]
            st.markdown("**Against a random article — is this a metric anything passes?**")
            g = st.columns(3)
            g[0].metric("Grounded, article it cited",
                        f"{100 * gn['grounded_rate_cited']:.1f}%",
                        delta=f"mean overlap {gn['mean_overlap_cited']:.0%}",
                        delta_color="off")
            g[1].metric("Grounded, random article",
                        f"{100 * gn['grounded_rate_random_same_window']:.1f}%",
                        delta=f"mean overlap {gn['mean_overlap_random_same_window']:.1%}",
                        delta_color="off")
            g[2].metric("Random draws", f"{gn['n_random_draws']:,}",
                        delta=f"over {gn['n_claims_scored']} claims", delta_color="off")
            st.caption(gn.get("statement", ""))

        if RETEST:
            st.subheader("Label stability")
            c = st.columns(3)
            c[0].metric("Agreement on 'confident'",
                        f"{RETEST['agreement_is_confident']:.0%}",
                        help=f"Cohen's kappa {RETEST['kappa_is_confident']}")
            c[1].metric("Agreement on driver identified",
                        f"{RETEST['agreement_driver_identified']:.0%}")
            c[2].metric("Exact confidence label",
                        f"{RETEST['agreement_exact_confidence_label']:.0%}")
            st.caption(
                "Sampling parameters were removed from the API, so identical inputs no "
                "longer guarantee identical outputs, and the placebo statistic is an "
                "LLM self-report. This is how much of it is stable signal."
            )


# ---------------------------------------------------------------------------
# Your data — the pipeline applied to any regime model
# ---------------------------------------------------------------------------

elif view == "Your data":
    from regime_narrative import pipeline as pl
    from regime_narrative import runs as runstore

    st.title("Run this on your own regime model")
    st.caption(
        "This pipeline knows nothing about HMMs. Give it dates from any regime "
        "model, any asset, any method — it retrieves a news window that provably "
        "closes on each date, explains it, and then measures whether the "
        "explanations mean anything."
    )

    st.markdown(
        '<div class="good"><b>What you get that you cannot get from just asking a '
        "model.</b> Anyone can ask an LLM what happened in March 2020. What is hard "
        "is knowing whether the answer is specific to those two weeks or generic "
        "commentary that would fit any month. Every run here comes back with a "
        "blind-matching score against chance, a citation-grounding rate, and an "
        "era-matched control arm — so the explanation arrives with its own error "
        "bars.</div>",
        unsafe_allow_html=True,
    )

    saved = runstore.list_runs()
    if saved:
        with st.expander(f"📂 Reload a previous run ({len(saved)} saved)", expanded=False):
            st.caption(
                "Runs are written to disk when they finish, so a refresh or a "
                "closed tab does not lose them. Individual narratives and news "
                "windows are cached separately, so re-running the same dates is "
                "free and near-instant either way."
            )
            pick = st.selectbox(
                "Saved run", saved, format_func=lambda r: r.display,
                label_visibility="collapsed",
            )
            c1, c2, c3 = st.columns([1, 1, 3])
            if c1.button("Load"):
                st.session_state["byo_loaded"] = runstore.load_run(pick.run_id)
                st.session_state.pop("byo_result", None)
                st.rerun()
            if c2.button("Delete"):
                runstore.delete_run(pick.run_id)
                st.rerun()
            c3.caption(f"`{pick.path.name}` · {pick.model} · "
                       f"{pick.window_days}-day windows · {pick.n_narratives} explanations")

    # A reloaded run is displayed from its file and short-circuits the rest.
    loaded = st.session_state.get("byo_loaded")
    if loaded and not st.session_state.get("byo_result"):
        st.success(f"Loaded **{loaded['label']}** — saved {loaded['saved_at'][:16].replace('T', ' ')}")
        r = loaded["retrieval"]
        c = st.columns(4)
        c[0].metric("Windows retrieved", f"{r['n_retrieved']}/{r['n_requested']}")
        c[1].metric("News items", f"{r['total_items']:,}")
        c[2].metric("Retrieval failures", r["n_failed"])
        c[3].metric("Post-boundary survivors",
                    "0" if not r["any_post_boundary_survivors"] else "LEAK")
        m = st.columns(3)
        lbm = loaded.get("blind_match", {}).get("transitions_all_sections", {})
        if "error" not in lbm and lbm:
            m[0].metric("Blind match", f"{lbm['n_correct']}/{lbm['n_scored']}",
                        delta=f"{100 * lbm['accuracy']:.0f}% vs {100 * lbm['chance']:.0f}% chance")
        lfa = loaded.get("faithfulness", {}).get("all", {})
        if lfa:
            m[1].metric("Claims grounded", f"{100 * lfa.get('grounding_rate', 0):.0f}%",
                        delta=f"{lfa.get('n_claims', 0)} claims", delta_color="off")
        lpb = loaded.get("placebo", {})
        if lpb.get("controls", {}).get("n"):
            m[2].metric("Confident: events vs controls",
                        f"{100 * lpb['transitions']['confident_rate']:.0f}% / "
                        f"{100 * lpb['controls']['confident_rate']:.0f}%",
                        delta=f"Fisher p = {lpb['fisher_p']:.3f}" if lpb.get("fisher_p") else None,
                        delta_color="off")
        st.dataframe(pd.DataFrame(loaded.get("records", [])), hide_index=True,
                     width='stretch', height=300)
        st.download_button(
            "Download this run (JSON)",
            json.dumps(loaded, indent=2).encode("utf-8"),
            file_name=f"{loaded['run_id']}.json", mime="application/json",
        )
        if st.button("Start a new run instead"):
            st.session_state.pop("byo_loaded", None)
            st.rerun()
        st.stop()

    st.subheader("1 · Your dates")
    tab_up, tab_paste, tab_demo = st.tabs(["Upload CSV", "Paste dates", "Use this study"])

    raw_df = None
    with tab_up:
        st.caption(
            "Any CSV with either **a column of transition dates**, or **a date "
            "column plus a state/regime column** (transitions are derived with a "
            "persistence filter). Column names are detected automatically."
        )
        up = st.file_uploader("CSV", type=["csv"], label_visibility="collapsed")
        if up is not None:
            raw_df = pd.read_csv(up)
    with tab_paste:
        txt = st.text_area(
            "One date per line (YYYY-MM-DD)", height=150,
            placeholder="2020-01-31\n2020-03-23\n2022-02-10",
        )
        if txt.strip():
            raw_df = pd.DataFrame({"date": [l.strip() for l in txt.splitlines() if l.strip()]})
    with tab_demo:
        st.caption("Load this study's 20 seed-stable transitions as a worked example.")
        if st.button("Load the SPY study"):
            raw_df = pd.DataFrame({"date": [t["date"] for t in TRANSITIONS]})
            st.session_state["byo_df"] = raw_df

    if raw_df is None and "byo_df" in st.session_state:
        raw_df = st.session_state["byo_df"]
    if raw_df is not None:
        st.session_state["byo_df"] = raw_df

    if raw_df is None:
        st.stop()

    try:
        dates, states, notes = pl.parse_input_frame(raw_df)
    except Exception as exc:
        st.error(f"Could not read that: {exc}")
        st.stop()
    for n in notes:
        st.caption(f"· {n}")

    st.subheader("2 · Settings")
    c = st.columns(4)
    win = c[0].number_input("News window (days)", 3, 45, int(CFG["news"]["window_days"]),
                            help="Long enough to contain the cause, short enough to "
                                 "stay specific.")
    dwell = c[1].number_input("Min dwell (days)", 1, 90,
                              int(CFG["transitions"]["min_dwell_days"]),
                              help="Only used when you supply a state series. A "
                                   "single-day flip is noise.",
                              disabled=states is None)
    n_ctrl = c[2].number_input("Controls per date", 0, 4,
                               int(CFG["placebo"]["n_per_transition"]),
                               help="Era-matched non-event dates. Zero skips the "
                                    "placebo arm — but then you cannot tell whether "
                                    "the explanations are diagnostic.")
    offmin = c[3].number_input("Control offset (days)", int(win) + 1, 400,
                               max(int(CFG["placebo"]["offset_days_min"]), int(win) + 1),
                               help="Must exceed the window length or control and "
                                    "event windows share articles.")

    if states is not None:
        dates = pl.derive_transitions(states, int(dwell))
        st.caption(f"· Derived **{len(dates)}** transitions at a {dwell}-day "
                   f"minimum dwell.")

    if not dates:
        st.warning("No dates to work with.")
        st.stop()

    cfg = pl.PipelineConfig.from_settings()
    cfg.window_days = int(win)
    cfg.placebo_per_transition = int(n_ctrl)
    cfg.placebo_offset_min = int(offmin)
    cfg.placebo_offset_max = max(int(offmin) + 60, cfg.placebo_offset_max)
    cfg.run_label = "byo"

    universe = {d.date() for d in states.index} if states is not None else None
    controls = pl.sample_controls(dates, universe=universe, cfg=cfg) if n_ctrl else []

    st.subheader("3 · What this will cost")
    est = pl.estimate_cost(len(dates) + len(controls), cfg)
    c = st.columns(5)
    c[0].metric("Event dates", len(dates))
    c[1].metric("Control dates", len(controls))
    c[2].metric("Wikipedia requests", f"{est['wikipedia_requests']:,}")
    c[3].metric("Estimated time", f"~{est['estimated_total_minutes']:.0f} min")
    c[4].metric("Estimated cost", f"~${est['estimated_usd_opus']:.2f}")
    st.caption(
        "Wikipedia is throttled to about one request a second by their policy, and "
        "that dominates the wall-clock, not the model. Cached windows and cached "
        "narratives are free and instant on a rerun."
    )

    if len(dates) < 5:
        st.warning(
            f"With {len(dates)} event dates the controls will be badly underpowered. "
            "Blind matching scores against 1/N chance, and the placebo comparison "
            "needs enough of both arms to resolve a difference. Ten or more is "
            "where the measurements start to mean something."
        )

    key_ready = has_env("ANTHROPIC_API_KEY")
    c1, c2 = st.columns([1, 3])
    dry = c1.checkbox("Retrieval only", value=not key_ready,
                      help="Fetch and check coverage without calling the model. "
                           "Free.")
    if not key_ready and not dry:
        st.warning("Set an API key in the sidebar, or tick 'Retrieval only'.")

    if st.button("Run", type="primary", disabled=(not key_ready and not dry)):
        bar = st.progress(0.0, text="Starting")

        def prog(msg, frac):
            bar.progress(frac, text=msg)

        with st.spinner("Working"):
            res = pl.run(dates, controls=controls, cfg=cfg,
                         generate=not dry, progress=prog)
        bar.empty()
        st.session_state["byo_result"] = res
        st.session_state.pop("byo_loaded", None)
        if res.narratives:
            try:
                rid = runstore.save_run(
                    res, cfg=cfg, model=CFG["llm"]["model"],
                    label=f"{len(res.transitions)} events",
                )
                st.success(f"Saved as `{rid}` — reload it from the panel above "
                           "even after a refresh.")
            except Exception as exc:
                st.warning(f"Run finished but could not be saved: {exc}")

    res = st.session_state.get("byo_result")
    if res is None:
        st.stop()

    st.subheader("4 · Results")
    r = res.retrieval
    c = st.columns(4)
    c[0].metric("Windows retrieved", f"{r['n_retrieved']}/{r['n_requested']}")
    c[1].metric("News items", f"{r['total_items']:,}")
    c[2].metric("Retrieval failures", r["n_failed"])
    c[3].metric("Post-boundary survivors",
                "0" if not r["any_post_boundary_survivors"] else "LEAK")

    for w in res.warnings[:8]:
        st.caption(f"⚠️ {w}")

    if not res.narratives:
        st.info("Retrieval only — no explanations generated.")
        st.stop()

    st.markdown("**Do the explanations mean anything?**")
    m = st.columns(3)
    bm = res.blind_match.get("transitions_all_sections", {})
    if "error" not in bm:
        m[0].metric("Blind match", f"{bm['n_correct']}/{bm['n_scored']}",
                    delta=f"{100 * bm['accuracy']:.0f}% vs {100 * bm['chance']:.0f}% chance",
                    help=f"permutation p = {bm['p_value']:.4g}")
    fa = res.faithfulness.get("all", {})
    if fa:
        m[1].metric("Claims grounded", f"{100 * fa.get('grounding_rate', 0):.0f}%",
                    delta=f"{fa.get('n_claims', 0)} claims audited", delta_color="off")
    p = res.placebo
    if p.get("controls", {}).get("n"):
        m[2].metric("Confident: events vs controls",
                    f"{100 * p['transitions']['confident_rate']:.0f}% / "
                    f"{100 * p['controls']['confident_rate']:.0f}%",
                    delta=f"Fisher p = {p['fisher_p']:.3f}" if p.get("fisher_p") else None,
                    delta_color="off")
        if p.get("fisher_p") and p["fisher_p"] > 0.05:
            st.markdown(
                '<div class="caveat"><b>The control arm did not separate.</b> The '
                "model produced confident explanations for non-event dates at a "
                "similar rate. With small samples this usually means the test was "
                "underpowered rather than that no difference exists — but either "
                "way, the explanations are not yet shown to be diagnostic of your "
                "regime changes. Report it.</div>",
                unsafe_allow_html=True,
            )
    else:
        st.caption("No control arm — set 'Controls per date' above 0 to run the "
                   "placebo comparison.")

    tbl = pl.to_records(res)
    st.dataframe(tbl, hide_index=True, width='stretch', height=320)

    d1, d2 = st.columns(2)
    d1.download_button(
        "Download results (CSV)", tbl.to_csv(index=False).encode("utf-8"),
        file_name="regime_narratives.csv", mime="text/csv",
    )
    payload = {
        "event_dates": [d.isoformat() for d in res.transitions],
        "control_dates": [d.isoformat() for d in res.controls],
        "settings": {"window_days": cfg.window_days,
                     "controls_per_date": cfg.placebo_per_transition,
                     "control_offset_min": cfg.placebo_offset_min,
                     "model": CFG["llm"]["model"],
                     "prompt_version": CFG["llm"]["prompt_version"]},
        "retrieval": res.retrieval,
        "placebo": res.placebo,
        "blind_match": {k: {kk: vv for kk, vv in v.items() if kk != "per_item"}
                        for k, v in res.blind_match.items() if isinstance(v, dict)},
        "faithfulness": {k: v for k, v in res.faithfulness.items() if k != "per_window"},
        "narratives": {k: n.as_dict() for k, n in res.narratives.items()},
        "warnings": res.warnings,
    }
    d2.download_button(
        "Download full results + provenance (JSON)",
        json.dumps(payload, indent=2, default=str).encode("utf-8"),
        file_name="regime_narratives.json", mime="application/json",
    )

    st.subheader("5 · Read them")
    pick = st.selectbox("Window", sorted(res.narratives),
                        format_func=lambda k: f"{res.windows[k]['kind']}  ·  "
                                              f"{res.windows[k]['date']}")
    n = res.narratives[pick]
    if not n.driver_identified:
        st.warning("**No identifiable driver.** The model declined.")
        if n.insufficient_evidence_reason:
            st.write(n.insufficient_evidence_reason)
    else:
        badge = {"high": "🟢 high", "medium": "🟡 medium",
                 "low": "🟠 low", "none": "⚪ none"}.get(n.confidence, n.confidence)
        st.markdown(f"**Confidence: {badge}**")
        st.write(n.summary)
        if n.mechanism:
            st.markdown(f"**Mechanism.** {n.mechanism}")
        if n.competing_explanation:
            with st.expander("Strongest competing explanation"):
                st.write(n.competing_explanation)
    for cl in n.claims:
        st.markdown(
            f"• {cl['claim']}<br><span class='idtag'>cites "
            f"{', '.join(cl.get('item_ids', [])) or 'nothing'}</span>",
            unsafe_allow_html=True,
        )


# ---------------------------------------------------------------------------
# Live
# ---------------------------------------------------------------------------

elif view == "Live":
    st.title("Live")
    st.caption(
        "Pick any date. The pipeline fetches a boundary-enforced news window, "
        "scrubs dates from it, and asks the model what happened — using the same "
        "prompt as every result in this report."
    )

    st.markdown(
        '<div class="good"><b>What this demonstrates.</b> The HMM has already decided '
        "which days are transitions; the language model never sees a price. Point it "
        "at an ordinary Tuesday and watch it decline — that is the placebo test, run "
        "one date at a time.</div>",
        unsafe_allow_html=True,
    )

    c = st.columns([1, 1, 1, 1])
    tdates = [t["date"] for t in TRANSITIONS]
    preset = c[0].selectbox("Jump to", ["(custom)"] + tdates)
    default = date.fromisoformat(preset) if preset != "(custom)" else date(2020, 1, 31)
    chosen = c[1].date_input("Boundary date", value=default,
                             min_value=date(2005, 1, 1), max_value=date.today())
    window_days = c[2].number_input("Window (days)", 3, 45,
                                    int(CFG["news"]["window_days"]))
    is_transition = chosen.isoformat() in tdates
    c[3].metric("HMM says", "TRANSITION" if is_transition else "no transition")

    if not has_env("ANTHROPIC_API_KEY"):
        st.warning("ANTHROPIC_API_KEY is not set — retrieval will work, generation will not.")

    go_btn = st.button("Run the pipeline", type="primary")

    if go_btn:
        from regime_narrative.narrative import (
            audit_date_leaks, generate_narrative, render_window,
        )
        from regime_narrative.news.wikipedia import fetch_window

        with st.status("Running", expanded=True) as status:
            st.write(f"Retrieving news, window closing **{chosen}** …")
            w = fetch_window(chosen, window_days=int(window_days),
                             max_items=int(CFG["news"]["max_items_per_window"]))
            st.write(
                f"✅ {len(w)} items · {w.provenance['n_pages_ok']}/{window_days} pages · "
                f"**{w.dropped_post_boundary}** items post-dated the boundary and were dropped"
            )
            text, _ = render_window(w)
            leaks = audit_date_leaks(text)
            total_leaks = sum(leaks.values())
            st.write(f"✅ Date scrub: **{total_leaks}** residual date signals in "
                     f"{len(text):,} characters")

            nar = None
            if has_env("ANTHROPIC_API_KEY"):
                st.write(f"Asking `{CFG['llm']['model']}` …")
                nar = generate_narrative(
                    w, window_id=f"live_{chosen.isoformat()}",
                    kind="transition" if is_transition else "placebo",
                    run_label="live",
                )
                st.write("✅ Response received")
            status.update(label="Done", state="complete", expanded=False)

        if nar:
            st.subheader("Explanation")
            if not nar.driver_identified:
                st.warning("**No identifiable driver.** The model declined.")
                if nar.insufficient_evidence_reason:
                    st.write(nar.insufficient_evidence_reason)
            else:
                badge = {"high": "🟢 high", "medium": "🟡 medium",
                         "low": "🟠 low", "none": "⚪ none"}.get(nar.confidence, nar.confidence)
                st.markdown(f"**Confidence: {badge}**")
                st.write(nar.summary)
                if nar.mechanism:
                    st.markdown(f"**Mechanism.** {nar.mechanism}")
                if nar.competing_explanation:
                    with st.expander("Strongest competing explanation"):
                        st.write(nar.competing_explanation)
            cited = set(nar.primary_item_ids or [])
            for cl in nar.claims:
                cited.update(cl.get("item_ids", []))
        else:
            cited = set()

        st.subheader(f"The window the model saw — {len(w)} items")
        st.caption(
            f"{w.start_instant:%Y-%m-%d} → {w.end_instant:%Y-%m-%d} UTC · "
            "revision-pinned · items the model cited are highlighted"
        )
        by_day: dict[str, list] = {}
        for it in w.items:
            by_day.setdefault(it.extra.get("event_date", "?"), []).append(it)
        for day, items in sorted(by_day.items()):
            with st.expander(f"Day {(date.fromisoformat(day) - w.start_instant.date()).days + 1}"
                             f"  ·  {len(items)} items"
                             + ("  ·  cited" if any(i.item_id in cited for i in items) else "")):
                for it in items:
                    klass = "newsitem cited" if it.item_id in cited else "newsitem"
                    st.markdown(
                        f"<div class='{klass}'><span class='idtag'>{it.item_id} · "
                        f"{it.extra.get('section', '')} · rev "
                        f"{it.extra.get('revision_id', '?')}</span><br>{it.text}</div>",
                        unsafe_allow_html=True,
                    )


# ---------------------------------------------------------------------------
# Provenance
# ---------------------------------------------------------------------------

elif view == "Provenance":
    st.title("Provenance")
    st.caption(
        "Every number traceable to the code that produced it, and every narrative "
        "traceable to the exact prompt and the exact news window behind it."
    )

    tab1, tab2, tab3 = st.tabs(["Prompt", "Settings", "Call log"])

    with tab1:
        from regime_narrative.narrative import load_prompt
        text, digest = load_prompt()
        st.caption(f"`prompts/{CFG['llm']['prompt_version']}.md` · sha256 `{digest}` · "
                   "used unchanged for transitions and controls alike")
        st.code(text, language="markdown")

    with tab2:
        st.caption("`settings.yaml` — nothing in `src/` hardcodes a value that lives here.")
        st.code((Path(__file__).parent / "settings.yaml").read_text(encoding="utf-8"),
                language="yaml")

    with tab3:
        log = Path("data/cache/llm_calls/calls.jsonl")
        if log.exists():
            rows = [json.loads(l) for l in log.read_text(encoding="utf-8").splitlines() if l.strip()]
            st.caption(f"{len(rows)} logged calls. Every one records the model id, "
                       "prompt hash, input hash, token counts and raw response.")
            df = pd.DataFrame(rows)
            keep = [c for c in ["timestamp_utc", "window_id", "kind", "run_label",
                                "model", "prompt_sha256_16", "input_hash",
                                "n_items_in_window", "usage"] if c in df.columns]
            st.dataframe(df[keep].tail(200), hide_index=True, width='stretch')
        else:
            st.info("No calls logged yet.")
