"""Build a deterministic synthetic operations fixture; no customer data."""
import csv,json,sqlite3
from pathlib import Path
from datetime import datetime,timedelta
ROOT=Path(__file__).resolve().parent
ASOF=datetime(2026,10,31)
rows=[]
for i in range(120):
    opened=datetime(2026,10,1)+timedelta(hours=i*5)
    reply=None if i%7==0 else opened+timedelta(hours=1+i%30)
    closed=None if i%4==0 else (reply or opened)+timedelta(hours=3+i%48)
    rows.append((i+1,['Email','Web','Phone'][i%3],['Delivery','Billing','Returns','General'][i%4],opened.isoformat(),reply.isoformat() if reply else None,closed.isoformat() if closed else None))
db=sqlite3.connect(':memory:')
db.executescript((ROOT/'schema.sql').read_text())
db.executemany('INSERT INTO enquiries VALUES(?,?,?,?,?,?)',rows)
assert len(rows)==len({r[0] for r in rows})==120
assert all((not r[4] or r[4]>=r[3]) and (not r[5] or r[5]>=r[3]) for r in rows)
with (ROOT/'enquiries.csv').open('w',newline='') as f:
    w=csv.writer(f);w.writerow(['enquiry_id','channel','category','opened_at','first_reply_at','resolved_at']);w.writerows(rows)
result=db.execute((ROOT/'metrics.sql').read_text(),{'asof':ASOF.isoformat()}).fetchone()
metrics=dict(zip(['enquiries','backlog','unanswered','mean_first_reply_hours','mean_resolution_hours','over_48h_open'],result))
assert metrics['backlog']==30 and metrics['unanswered']==18
assert abs(metrics['mean_first_reply_hours']-sum((datetime.fromisoformat(r[4])-datetime.fromisoformat(r[3])).total_seconds()/3600 for r in rows if r[4])/102)<1e-7
(ROOT/'metrics.json').write_text(json.dumps({'synthetic':True,'as_of':ASOF.isoformat(),'metrics':metrics},indent=2)+'\n')
channel=db.execute('SELECT channel,COUNT(*),SUM(resolved_at IS NULL) FROM enquiries GROUP BY channel').fetchall()
with (ROOT/'channel_summary.csv').open('w',newline='') as f:
    w=csv.writer(f);w.writerow(['channel','enquiries','backlog']);w.writerows(channel)
svg='<svg xmlns="http://www.w3.org/2000/svg" width="900" height="360"><rect width="900" height="360" fill="#f4f5f7"/><g font-family="Arial" fill="#172238"><text x="30" y="40" font-size="24">Synthetic enquiry workload by channel</text>'
for n,(name,total,backlog) in enumerate(channel):
    y=85+n*75
    svg+=f'<text x="30" y="{y+20}">{name}</text><rect x="120" y="{y}" width="{total*12}" height="28" fill="#2368a0"/><rect x="120" y="{y+30}" width="{backlog*12}" height="14" fill="#b57520"/><text x="630" y="{y+20}">{total} enquiries / {backlog} open</text>'
svg+='<text x="30" y="335">Blue: enquiries · amber: backlog · fictional fixture, not business performance</text></g></svg>'
(ROOT/'workload.svg').write_text(svg)
print(json.dumps(metrics,indent=2))
