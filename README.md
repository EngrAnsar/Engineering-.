# Electrical Engineering AI Copilot

A demo Streamlit agent for government electrical engineers (DISCO /
distribution & feeder scheme construction wings). Shows how AI can take
over repetitive information-processing work around an electrical scheme -
document extraction, BOQ/material rate checks, progress variance,
inspection support, report drafting, and delay-risk scoring - while every
output stays a **draft** for the engineer to review and sign.

Same architecture as the civil-works version it's based on: one Groq LLM
call per module, deterministic pre-processing wherever a wrong number
would matter (rate flagging, delay regression), and a system prompt per
tab that pins the model to "explain, don't decide."

## Modules

1. **Document Analyzer** - extracts project identity, contract terms,
   scope, milestones, and safety/clearance requirements from a pasted
   project brief.
2. **BOQ / Material Assistant** - flags BOQ line items whose rate falls
   outside an engineer-set typical range (deterministic Python), then
   asks the LLM to explain each flag and suggest one verification step.
3. **Progress Monitor** - planned vs. actual progress, activity-level
   chart, and an AI read of the contractor's monthly report (bottlenecks,
   follow-up questions for the SDO).
4. **Inspection Assistant** - turns rough field notes into a structured
   record, with safety observations (clearance, earthing, damaged
   insulators/fuses, unlocked panels) surfaced first.
5. **Report Generator** - stitches together whatever's already been
   generated (or falls back to the raw sample files) into a one-page
   draft progress report with a sign-off block.
6. **Delay Risk (ML)** - a small `scikit-learn` linear regression trained
   live on a historical-projects CSV predicts delay in months; the LLM
   explains the score and its limits in plain language.

## Local setup

```bash
pip install -r requirements.txt
cp .env.example .env   # then paste your Groq API key into .env
streamlit run app.py
```

Get a free Groq API key at https://console.groq.com/keys — the app also
lets you paste a key directly into the sidebar for a session without an
`.env` file.

## Deploying on Streamlit Community Cloud

1. Push this folder to a public (or private, if you have Streamlit Cloud
   access to private repos) GitHub repository. Keep `sample_data/` in the
   repo - the app reads its sample files from there.
2. **Do not commit your `.env` file.** A `.gitignore` entry is included.
3. Go to https://share.streamlit.io, sign in, and click "New app."
4. Point it at your repo, branch, and `app.py` as the entry point.
5. In the app's **Settings → Secrets**, add:
   ```toml
   GROQ_API_KEY = "your_key_here"
   ```
   Streamlit Cloud exposes `st.secrets` values as environment variables
   automatically for apps that read `os.environ`, but to be safe you can
   also add one line near the top of `app.py` if needed:
   ```python
   os.environ.setdefault("GROQ_API_KEY", st.secrets.get("GROQ_API_KEY", ""))
   ```
6. Deploy. Anyone using the live app can also just paste their own key in
   the sidebar - it's never persisted server-side, only kept in that
   session's `st.session_state`.

## Customizing for a different scheme

Replace the five files in `sample_data/` with your own case file (project
brief, BOQ, progress report, inspection notes, historical projects), and
adjust `BOQ_TYPICAL_RANGE` in `app.py` to match your own BOQ item numbers
and rate bands. Everything else (prompts, tab structure, regression
model) will keep working as-is.

## Design principle

Every system prompt in this app repeats a version of the same
instruction on purpose: the model explains, drafts, and flags - it never
certifies, energizes, approves, or decides. That's not boilerplate; it's
the actual safety boundary of the tool. Keep it when you adapt this for
your own scheme.
