# Review and reproduction checklist

Use the standard-library pattern matcher to review the enquiry workflow without the optional neural dependencies. It is exact intent matching, not a validated language model. The neural path is historical and unverified in this review. The HubSpot adapter has mock coverage, not a confirmed live connection. Use fictional demonstration data; no customer workload or time-saving result is claimed.

- Follow the repository’s existing setup instructions and pinned reproduction environment where provided.
- Inspect the source manifest and compare generated metrics with the documented baseline.
- Use original artefacts only; label screenshots by application, filter context and verification status.
- Keep proposal, observed association and implemented outcome separate.
- Preserve source attribution, original files and assisted-development disclosures.
- Recover missing source material before strengthening claims.

This documentation update does not change analysis code, dependencies, data, native report files or model results.

## Windows connection cleanup

The local review found that SQLite transaction contexts committed or rolled back without closing the connection. On Windows, this prevented temporary database cleanup. `crm.database` now closes connections after either outcome. The added regression check verifies commit cleanup, rollback cleanup and rollback contents. All 11 standard-library CRM tests pass; this does not validate the optional neural model or a live HubSpot connection.
