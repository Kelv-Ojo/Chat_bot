"""Reproducible fictional workload, with SQL totals reconciled independently."""
import argparse
import json
from collections import Counter
from pathlib import Path
import bot_engine
import crm

EXAMPLES = [
    ('Alex Demo', 'alex@example.com', 'Do you offer international shipping?'),
    ('Alex Demo', 'alex@example.com', 'What is your price matching policy?'),
    ('Sam Sample', 'sam@example.com', 'Can I cancel my order?'),
    ('Sam Sample', 'sam@example.com', 'zqx obscure enquiry'),
    ('Jo Example', 'jo@example.com', 'Do you offer gift wrapping services?'),
    ('Jo Example', 'jo@example.com', 'How can I contact customer support?'),
    ('Lee Test', 'lee@example.com', 'Can I request an invoice for my order?'),
    ('Lee Test', 'lee@example.com', 'Please investigate my unusual account problem'),
]


def run(path, report_path):
    crm.initialise(path)
    for i, (name, email, message) in enumerate(EXAMPLES):
        crm.capture(path, name, email, message, bot_engine.demo_answer(message), f'fixture-{i}')
    # Use a dedicated demo DB: these operations affect fixture records only.
    with crm.database(path) as db:
        first = db.execute("SELECT id FROM enquiries WHERE request_id='fixture-0'").fetchone()['id']
    crm.set_status(path, first, 'Resolved')
    records = crm.rows(path)
    m = crm.metrics(path)
    assert m['enquiries'] == len(records)
    assert m['customers'] == len({r['contact_id'] for r in records})
    assert m['needs_review'] == sum(r['needs_review'] for r in records)
    assert m['open'] + m['resolved'] == m['enquiries']
    patterns = {}
    for intent in bot_engine.intents():
        for pattern in intent['patterns']:
            patterns.setdefault(bot_engine.normalise(pattern), set()).add(intent['tag'])
    report = {'data': 'Fictional demonstration; not an AI accuracy evaluation or customer outcome study',
              'engine': 'demo-exact-match', 'metrics': m,
              'intent_counts': dict(Counter(r['intent'] for r in records)),
              'source_quality': {
                  'intent_entries': len(bot_engine.intents()),
                  'unique_intent_labels': len({i['tag'] for i in bot_engine.intents()}),
                  'conflicting_normalised_patterns': sum(len(labels) > 1 for labels in patterns.values())}}
    Path(report_path).parent.mkdir(parents=True, exist_ok=True)
    Path(report_path).write_text(json.dumps(report, indent=2) + '\n')
    print(json.dumps(report, indent=2))


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--db', default='data/demo.sqlite3')
    parser.add_argument('--report', default='reports/demo-report.json')
    args = parser.parse_args()
    run(args.db, args.report)

