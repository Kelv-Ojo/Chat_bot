"""Run: streamlit run app.py --server.address 127.0.0.1"""
import os
import sqlite3
from pathlib import Path
from uuid import uuid4
import streamlit as st
import bot_engine
import crm
from hubspot import HubSpotError, sync_contact

st.set_page_config(page_title='Enquiry Desk', page_icon='✳', layout='wide')
DB = Path(os.environ.get('CRM_DB_PATH', str(Path(__file__).parent / 'data' / 'crm.sqlite3')))
crm.initialise(DB)
st.title('Enquiry Desk')
st.caption('Chatbot → customer records → follow-up. A personal portfolio prototype using fictional data.')
page = st.radio('Workspace', ['New enquiry', 'CRM workspace', 'Reporting', 'HubSpot connection'], horizontal=True)

if page == 'New enquiry':
    st.subheader('Capture an enquiry')
    st.info('Use fictional names and example.com emails. Store policies in the original chatbot are illustrative. Replies are suggestions, not confirmed order actions.')
    engine = st.selectbox('Reply engine', ['Demo pattern matcher', 'Original neural model'])
    st.caption('Demo mode uses exact matches from the original intents; it is not AI inference. The neural model needs the optional model dependencies.')
    if 'request_id' not in st.session_state:
        st.session_state.request_id = str(uuid4())
    with st.form('enquiry'):
        name = st.text_input('Customer name', value='Alex Demo')
        email = st.text_input('Customer email', value='alex@example.com')
        message = st.text_area('Enquiry', placeholder='Do you offer international shipping?', max_chars=2000)
        handoff = st.checkbox('Customer requests human follow-up')
        submitted = st.form_submit_button('Save enquiry', type='primary')
    if submitted:
        try:
            if not message.strip():
                raise ValueError('Enter an enquiry first.')
            result = bot_engine.demo_answer(message) if engine.startswith('Demo') else bot_engine.neural_answer(message)
            if handoff:
                result = bot_engine.Result(result.intent, result.reply, result.engine, result.score, True)
            enquiry_id = crm.capture(DB, name, email, message, result, st.session_state.request_id)
            st.session_state.request_id = str(uuid4())
            st.success(f'Enquiry #{enquiry_id} saved to the local CRM. Status: New.')
            st.write(result.reply)
            st.caption(f'Intent: {result.intent} · Human review: {"yes" if result.needs_review else "not flagged"}. An answer suggestion does not close an enquiry.')
        except (ValueError, sqlite3.Error) as exc:
            st.error(str(exc))
        except (ImportError, LookupError, OSError):
            st.error('Original model unavailable. Follow MODEL_SETUP.md or choose Demo pattern matcher. Nothing was saved.')

elif page == 'CRM workspace':
    st.subheader('Enquiries and follow-up')
    records = crm.rows(DB)
    status_filter = st.selectbox('Filter by status', ['All', *crm.STATUSES])
    shown = [r for r in records if status_filter == 'All' or r['status'] == status_filter]
    columns = ['id', 'name', 'intent', 'needs_review', 'status', 'created_at']
    st.dataframe([{key: row[key] for key in columns} for row in shown], use_container_width=True, hide_index=True)
    if shown:
        selected = st.selectbox('Enquiry to update', [r['id'] for r in shown])
        current = next(r for r in shown if r['id'] == selected)
        st.write(f"**Customer:** {current['name']} · {current['email']}")
        st.text(f"Enquiry: {current['message']}\nSuggested reply: {current['reply']}")
        st.caption(f"Reply engine: {current['engine']}. Review flag: {bool(current['needs_review'])}.")
        status = st.selectbox('Status', crm.STATUSES, index=crm.STATUSES.index(current['status']))
        if st.button('Update status'):
            crm.set_status(DB, selected, status)
            st.rerun()
    st.subheader('Customer records')
    st.dataframe(crm.contacts(DB), use_container_width=True, hide_index=True)

elif page == 'Reporting':
    st.subheader('Service workload')
    m = crm.metrics(DB)
    cols = st.columns(4)
    for col, label, key in zip(cols, ['Enquiries', 'Customers', 'Open enquiries', 'Flagged for review'], ['enquiries', 'customers', 'open', 'needs_review']):
        col.metric(label, m[key])
    rate = 'No observations' if m['review_rate'] is None else f"{m['review_rate']:.1%}"
    st.write(f'Human-review flag rate: {rate}')
    st.caption('Denominator: all captured enquiries, including resolved ones. A flag is routing metadata, not a failure or measured resolution. No time savings or accuracy claims are made.')
    records = crm.rows(DB)
    if records:
        counts = {}
        for record in records:
            counts[record['intent']] = counts.get(record['intent'], 0) + 1
        st.bar_chart(counts)
    st.code('SELECT intent, COUNT(*) AS enquiries FROM enquiries GROUP BY intent ORDER BY enquiries DESC;', language='sql')

else:
    st.subheader('HubSpot contact connection')
    st.write('This optional step finds or creates a fictional contact in your HubSpot test account. It sends the selected name and email only. Enquiries remain local; tickets and notes are not synced.')
    st.caption('Keep HUBSPOT_ACCESS_TOKEN in your local environment. Never paste it into GitHub or this chat. See HUBSPOT_SETUP.md.')
    available = crm.contacts(DB)
    if not available:
        st.info('Save a fictional enquiry first.')
    else:
        chosen = st.selectbox('Contact to sync', [c['id'] for c in available])
        contact = next(c for c in available if c['id'] == chosen)
        st.json({'name': contact['name'], 'email': contact['email']})
        approved = st.checkbox('I authorise sending this fictional contact to my HubSpot test account')
        token = os.environ.get('HUBSPOT_ACCESS_TOKEN', '')
        if not token:
            st.info('Not connected. Local CRM features work without HubSpot.')
        if st.button('Find or create HubSpot contact', disabled=not (approved and token)):
            try:
                result = sync_contact(token, contact['name'], contact['email'])
                st.success(f"HubSpot confirmed contact {result['id']} ({result['action']}).")
            except (ValueError, HubSpotError) as exc:
                st.error(str(exc))

