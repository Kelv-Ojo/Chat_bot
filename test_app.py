"""Exercise the real Streamlit form, persistence, status update and reporting."""
import os
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch
from streamlit.testing.v1 import AppTest
import crm


class AppTests(unittest.TestCase):
    def test_capture_follow_up_reporting_and_disconnected_sync(self):
        with tempfile.TemporaryDirectory() as folder, patch.dict(os.environ, {
            'CRM_DB_PATH': str(Path(folder) / 'ui.sqlite3'), 'HUBSPOT_ACCESS_TOKEN': ''
        }):
            app = AppTest.from_file('app.py', default_timeout=30).run()
            self.assertEqual(len(app.exception), 0)
            app.text_area[0].set_value('Do you offer international shipping?')
            app.button[0].click().run()
            self.assertEqual(len(app.exception), 0)
            self.assertIn('saved to the local CRM', app.success[0].value)
            app.text_area[0].set_value('a question that has never been seen before')
            app.button[0].click().run()
            db = Path(folder) / 'ui.sqlite3'
            self.assertEqual(len(crm.contacts(db)), 1)
            self.assertEqual(crm.metrics(db)['enquiries'], 2)
            app.radio[0].set_value('CRM workspace').run()
            self.assertEqual(len(app.exception), 0)
            app.selectbox[2].set_value('Resolved')
            app.button[0].click().run()
            self.assertEqual(crm.metrics(db)['resolved'], 1)
            app.radio[0].set_value('Reporting').run()
            self.assertEqual(len(app.exception), 0)
            self.assertEqual([m.value for m in app.metric], ['2', '1', '1', '1'])
            app.radio[0].set_value('HubSpot connection').run()
            self.assertEqual(len(app.exception), 0)
            self.assertTrue(app.button[0].disabled)


if __name__ == '__main__':
    unittest.main()
