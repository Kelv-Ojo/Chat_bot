"""Snapshot regression tests against executable SQL, with hand-calculated expectations."""
import sqlite3
import unittest
from pathlib import Path
ROOT=Path(__file__).resolve().parent/'operations'
class SnapshotTests(unittest.TestCase):
 def metrics(self,rows,asof='2026-10-02T00:00:00'):
  with sqlite3.connect(':memory:') as db:
   db.executescript((ROOT/'schema.sql').read_text())
   db.executemany('INSERT INTO enquiries VALUES(?,?,?,?,?,?)',rows)
   return db.execute((ROOT/'metrics.sql').read_text(),{'asof':asof}).fetchone()
 def row(self,opened='2026-10-01T00:00:00',reply=None,closed=None):
  return (1,'Email','General',opened,reply,closed)
 def test_future_events_remain_open_and_unanswered(self):
  self.assertEqual(self.metrics([self.row(reply='2026-10-03T00:00:00',closed='2026-10-04T00:00:00')]),(1,1,1,None,None,0))
 def test_future_opening_excluded(self):
  self.assertEqual(self.metrics([self.row(opened='2026-10-03T00:00:00')]),(0,0,0,None,None,0))
 def test_events_exactly_at_snapshot_included(self):
  m=self.metrics([self.row(reply='2026-10-02T00:00:00',closed='2026-10-02T00:00:00')])
  self.assertEqual(m,(1,0,0,24.0,24.0,0))
 def test_empty_snapshot(self):
  self.assertEqual(self.metrics([]),(0,0,0,None,None,0))
 def test_age_exactly_48_hours_is_not_overdue(self):
  self.assertEqual(self.metrics([self.row()],asof='2026-10-03T00:00:00')[-1],0)
 def test_age_over_48_hours_is_overdue(self):
  self.assertEqual(self.metrics([self.row()],asof='2026-10-03T00:00:01')[-1],1)
 def test_missing_reply_excluded_from_average(self):
  a=self.row(reply='2026-10-01T12:00:00');b=(2,'Web','General','2026-10-01T00:00:00',None,None)
  m=self.metrics([a,b]);self.assertEqual(m[:3],(2,2,1));self.assertAlmostEqual(m[3],12)
 def test_future_resolution_does_not_affect_answered_duration(self):
  m=self.metrics([self.row(reply='2026-10-01T06:00:00',closed='2026-10-04T00:00:00')]);self.assertEqual(m,(1,1,0,6.0,None,0))
if __name__=='__main__':unittest.main()
