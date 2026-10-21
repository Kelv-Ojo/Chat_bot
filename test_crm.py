import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch
from urllib.error import HTTPError
import bot_engine
import crm
import hubspot


class CRMTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.db = Path(self.temp.name) / 'test.sqlite3'
        crm.initialise(self.db)
        self.result = bot_engine.demo_answer('Do you offer international shipping?')

    def tearDown(self):
        self.temp.cleanup()

    def test_contact_deduplication_and_idempotent_submission(self):
        first = crm.capture(self.db, 'Alex Demo', ' ALEX@EXAMPLE.COM ', 'first', self.result, 'one')
        duplicate = crm.capture(self.db, 'Alex Demo', 'alex@example.com', 'first', self.result, 'one')
        crm.capture(self.db, 'Alex Demo', 'alex@example.com', 'second', self.result, 'two')
        self.assertEqual(first, duplicate)
        self.assertEqual(len(crm.contacts(self.db)), 1)
        self.assertEqual(crm.metrics(self.db)['enquiries'], 2)

    def test_validation_does_not_create_partial_contact(self):
        for name, email, message in [('', 'alex@example.com', 'Hi'),
                                     ('Alex', 'bad email', 'Hi'), ('Alex', 'a@example.com', ' ')]:
            with self.assertRaises(ValueError):
                crm.capture(self.db, name, email, message, self.result, 'invalid')
        self.assertEqual(crm.contacts(self.db), [])

    def test_unknown_and_ambiguous_inputs_require_review(self):
        self.assertTrue(bot_engine.demo_answer('zqx obscure enquiry').needs_review)
        self.assertEqual(bot_engine.demo_answer('Can I cancel my order?').intent, 'ambiguous')
        self.assertFalse(self.result.needs_review)
        self.assertIsNone(self.result.score)

    def test_status_and_independent_metrics(self):
        first = crm.capture(self.db, 'Alex', 'alex@example.com', 'a', self.result, 'one')
        unknown = bot_engine.demo_answer('unseen question')
        crm.capture(self.db, 'Sam', 'sam@example.com', 'b', unknown, 'two')
        crm.set_status(self.db, first, 'Resolved')
        records = crm.rows(self.db)
        metrics = crm.metrics(self.db)
        self.assertEqual(metrics['open'], sum(r['status'] != 'Resolved' for r in records))
        self.assertEqual(metrics['review_rate'], .5)
        self.assertIsNotNone(next(r for r in records if r['id'] == first)['resolved_at'])
        crm.set_status(self.db, first, 'In progress')
        self.assertIsNone(next(r for r in crm.rows(self.db) if r['id'] == first)['resolved_at'])
        with self.assertRaises(ValueError):
            crm.set_status(self.db, first, 'invalid')
        with self.assertRaises(ValueError):
            crm.set_status(self.db, 999, 'New')

    def test_empty_metrics_and_sql_parameters(self):
        self.assertIsNone(crm.metrics(self.db)['review_rate'])
        crm.capture(self.db, "O'Example", 'alex@example.com', "'; DROP TABLE contacts;--", self.result, 'sql')
        self.assertEqual(len(crm.contacts(self.db)), 1)


class HubSpotTests(unittest.TestCase):
    @patch('hubspot._request')
    def test_existing_contact_is_not_overwritten(self, request):
        request.return_value = {'id': '123'}
        self.assertEqual(hubspot.sync_contact('test-token', 'Alex Demo', 'alex@example.com'),
                         {'id': '123', 'action': 'existing'})
        self.assertEqual(request.call_count, 1)
        self.assertEqual(request.call_args.args[1], 'GET')

    @patch('hubspot._request')
    def test_create_only_after_not_found(self, request):
        request.side_effect = [None, {'id': '456'}]
        self.assertEqual(hubspot.sync_contact('test-token', 'Alex Demo', 'alex@example.com')['action'], 'created')
        payload = request.call_args.args[3]
        self.assertEqual(payload['properties']['email'], 'alex@example.com')
        self.assertEqual(request.call_args.args[1], 'POST')

    @patch('hubspot._request')
    def test_uncertain_create_is_reconciled(self, request):
        request.side_effect = [None, hubspot.HubSpotError('timeout'), {'id': '789'}]
        self.assertEqual(hubspot.sync_contact('test-token', 'Alex Demo', 'alex@example.com')['id'], '789')
        self.assertEqual([call.args[1] for call in request.call_args_list], ['GET', 'POST', 'GET'])

    @patch('hubspot._request')
    def test_real_contact_and_missing_token_are_rejected(self, request):
        with self.assertRaises(ValueError):
            hubspot.sync_contact('test-token', 'Alex', 'alex@real-company.invalid')
        with self.assertRaises(ValueError):
            hubspot.sync_contact('', 'Alex', 'alex@example.com')
        request.assert_not_called()

    @patch('hubspot.urlopen')
    def test_http_error_is_sanitised(self, opener):
        opener.side_effect = HTTPError('https://api.hubapi.com', 401, 'private-response', {}, None)
        with self.assertRaises(hubspot.HubSpotError) as caught:
            hubspot._request('secret-token', 'GET', '/crm/v3/objects/contacts/1')
        self.assertNotIn('secret-token', str(caught.exception))
        self.assertIn('401', str(caught.exception))


if __name__ == '__main__':
    unittest.main()

