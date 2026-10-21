"""Optional contact-only HubSpot integration. No network call on import."""
import json
from urllib.error import HTTPError, URLError
from urllib.parse import quote
from urllib.request import Request, urlopen


class HubSpotError(Exception):
    pass


def _request(token, method, path, payload=None):
    request = Request('https://api.hubapi.com' + path,
        data=json.dumps(payload).encode() if payload is not None else None,
        headers={'Authorization': 'Bearer ' + token, 'Content-Type': 'application/json'},
        method=method)
    try:
        with urlopen(request, timeout=15) as response:
            return json.load(response)
    except HTTPError as exc:
        if exc.code == 404:
            return None
        raise HubSpotError(f'HubSpot returned HTTP {exc.code}. Check access scopes, account and rate limits.') from None
    except (URLError, TimeoutError, ValueError):
        raise HubSpotError('HubSpot could not confirm the request. Check the account before retrying.') from None


def sync_contact(token, name, email):
    """Find by email; create only if absent. Never overwrite an existing contact."""
    email = email.strip().lower()
    if not token.strip():
        raise ValueError('Set HUBSPOT_ACCESS_TOKEN locally first.')
    if email.rsplit('@', 1)[-1] not in {'example.com', 'example.org', 'example.net'}:
        raise ValueError('This portfolio prototype only syncs fictional example-domain contacts.')
    path = '/crm/v3/objects/contacts/' + quote(email, safe='') + '?idProperty=email'
    existing = _request(token, 'GET', path)
    if existing:
        return {'id': existing['id'], 'action': 'existing'}
    first, _, last = name.strip().partition(' ')
    try:
        created = _request(token, 'POST', '/crm/v3/objects/contacts',
            {'properties': {'email': email, 'firstname': first, 'lastname': last}})
    except HubSpotError:
        # A concurrent create or uncertain response may already have made the contact.
        existing = _request(token, 'GET', path)
        if existing:
            return {'id': existing['id'], 'action': 'existing'}
        raise
    if not created or 'id' not in created:
        raise HubSpotError('HubSpot did not confirm a contact ID; inspect the account before retrying.')
    return {'id': created['id'], 'action': 'created'}

