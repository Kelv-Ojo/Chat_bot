# Enquiry Desk — chatbot to CRM

A personal portfolio prototype extending [Kelvin's original e-commerce chatbot](https://github.com/Kelv-Ojo/Chat_bot).
Built with Codex assistance, preserving the original intent dataset and providing
an optional adapter for its existing TensorFlow model. All example customers and
workload figures are fictional. No commercial deployment or time savings are claimed.

## The question

How can a chatbot leave a useful customer-service record rather than an isolated
answer? Capture the customer and enquiry, suggest a reply, route unclear questions
for review and support follow-up reporting.

**Workflow:** enquiry → intent/reply suggestion → SQLite CRM → human status update
→ workload report. An optional HubSpot adapter finds or creates a contact only.

## Run locally

Use **Python 3.11 or 3.12**, not the older Python supplied with macOS.

```sh
cd /path/to/chatbot-crm
python3.12 -m venv .venv
source .venv/bin/activate
python -m pip install -r requirements-crm.txt
python demo.py
CRM_DB_PATH=data/demo.sqlite3 python -m streamlit run app.py --server.address 127.0.0.1
```

On Kelvin's Mac, the existing environment can also run it:

```sh
cd "/Users/kelvino/Documents/Codex/2026-10-18/create-a-scheduled-task-called-weekday/outputs/chatbot-crm"
CRM_DB_PATH=data/demo.sqlite3 /opt/anaconda3/bin/python3 -m streamlit run app.py --server.address 127.0.0.1
```

Open the URL printed by Streamlit. Inspect **CRM workspace**, then submit a question.
Try `Do you offer international shipping?`, `Can I cancel my order?` and an unseen question.

The default **Demo pattern matcher** uses exact normalised matches from the original
patterns; it is not AI inference. It runs without TensorFlow. [MODEL_SETUP.md](MODEL_SETUP.md)
describes the separate original-model path and its unverified runtime status.
The old requirements file is retained for provenance; use `requirements-crm.txt` here.

## Data model and decisions

| Table | Grain | Key |
|---|---|---|
| contacts | One customer email | Lowercase, stripped email is unique |
| enquiries | One saved customer message | Unique submission UUID prevents retry duplication |

Repeat customers get another enquiry, not another contact. A transaction prevents
failed enquiry inserts from leaving partial contact updates. SQL parameters handle
user text safely. Original messages/replies, engine and human-review flags are retained.
Every enquiry starts **New**, even with an answer suggestion. Only an explicit status
change resolves it.

Email is a demo identity rule: shared or changed emails are not handled. There is
no login, permissions system, verified identity or order access. Run on localhost
with fictional data. This is not a production customer portal.

## Reproduce the checks and report

```sh
python -m unittest -v test_crm test_app
python demo.py --db data/demo.sqlite3 --report reports/demo-report.json
```

`demo.EXAMPLES` contains eight selected fictional messages, not a random sample.
Re-running it does not insert duplicate enquiries. Use its dedicated database so
manual test records do not mix with the fixture.

Verified fixture results:

| Measure | Result | Interpretation |
|---|---:|---|
| Enquiries | 8 | Eight saved fictional messages |
| Customer emails | 4 | Two messages per fictional customer |
| Review flags | 4 / 8 (50%) | Two unknown and two ambiguous pattern matches |
| Open enquiries | 7 | One example explicitly marked resolved |
| Resolved enquiries | 1 | Scripted status change, not confirmed customer resolution |

SQL totals reconcile against independently counted joined records in `demo.py`.
The flag-rate denominator includes all enquiries, including resolved ones. No
observations produces “No observations”, not a misleading 0%.

```sql
SELECT intent, COUNT(*) AS enquiries
FROM enquiries
GROUP BY intent
ORDER BY enquiries DESC;
```

The original data has **97 intent entries, 93 unique labels and 6 normalised
patterns associated with more than one label**. Demo mode routes conflicts for
review. The optional neural path retains the original model's classification;
these findings do not establish that model's error rate.

**Recommendation:** reconcile overlapping labels, then evaluate unseen paraphrases
before considering automatic responses. Measure actual handling time and confirmed
outcomes before claiming efficiency gains. This fixture is not an accuracy benchmark.

## HubSpot status

The contact adapter is implemented and mock-tested. **No HubSpot account has been
connected and no live contact has been created by this build.** [HUBSPOT_SETUP.md](HUBSPOT_SETUP.md)
explains scopes, local token storage and the manual confirmation flow.

It looks up contacts by email and creates only if absent; existing contacts remain
unchanged. It checks for an existing record after an uncertain create response.
It sends fictional name/email only, not enquiries, notes or tickets.

## Verification and learning

- Core tests cover deduplication, idempotency, invalid input, SQL text handling,
  empty metrics, status changes/reopening and independently reconciled totals.
- HubSpot mocks cover lookup, create, uncertain create, contact restrictions and
  sanitised HTTP errors. They do not establish live API compatibility.
- The Streamlit interaction test submits two enquiries, reuses a contact, changes
  status, checks reporting and verifies that live sync is disabled without a token.
- Original neural runtime and live HubSpot round trip remain unverified.

Use [LEARNING_GUIDE.md](LEARNING_GUIDE.md), make a change and explain the decisions
before presenting the work. This is an in-progress personal project, not prior
professional CRM-building experience.

Original documentation: [README.original.md](README.original.md). Original data and
model: `Kelv-Ojo/Chat_bot`. The original repository has no explicit licence file;
no new licence is asserted over its existing assets.
