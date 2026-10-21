# Optional HubSpot contact integration

Status: implemented and tested against mocked API responses; **not live-connected**.
The local CRM works without this step.

1. Use a dedicated HubSpot developer test account with fictional data.
2. Create an app/access token using HubSpot's current developer interface. Grant
   only `crm.objects.contacts.read` and `crm.objects.contacts.write`.
3. Put the token in your local shell environment as `HUBSPOT_ACCESS_TOKEN`.
   Never put it in source code, a screenshot, a committed file or a chat message.
4. Start the app from the same shell. Save `Alex Demo / alex@example.com` locally.
5. Open **HubSpot connection**. Review the name and email preview, tick the
   authorisation box, then click **Find or create HubSpot contact**.
6. Verify the contact in HubSpot. Repeat once: it should find the existing ID.

The implementation uses HubSpot's documented v3 Contacts endpoints:

- `GET /crm/v3/objects/contacts/{email}?idProperty=email`
- `POST /crm/v3/objects/contacts` with email, firstname and lastname.

Existing contacts are left unchanged. Names are split at the first space for the
new contact's firstname/lastname fields; that shortcut is for fictional demo names.
Only `example.com`, `example.org` and `example.net` contacts can be synced.
There is no marketing opt-in, email send, ticket creation, note sync or order access.
Do not test in a production account. Timeouts may occur after a server has accepted
a create; the client checks for an existing email before any later create attempt.

Reference: [HubSpot's official Contacts guide](https://developers.hubspot.com/blog/a-developers-guide-to-hubspot-crm-objects-contacts-object).
Account setup, permissions and a live create/find round trip remain to be verified
by Kelvin once an account is available.

