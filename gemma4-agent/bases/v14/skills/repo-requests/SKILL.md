---
name: repo-requests
description: Map of the psf/requests repository (the Requests HTTP library (sessions, adapters, models, cookies, auth, utils)). Load it first when the task's repository is psf/requests; it says which file holds each class or function so you can go straight to read_file.
---

Repository `psf/requests`: the Requests HTTP library (sessions, adapters, models, cookies, auth, utils). Source package: `src/requests/`. Tests: `tests/` (9 `test_*.py` files at the top level).

Use this map to pick the file, then confirm with one scoped `git grep -n "<symbol>" src/requests/` and `read_file` on that file only.
Do not grep the whole repository, docs or examples. The map was taken from one snapshot: when a name is missing, the code moved; fall back to `git grep`.

Files most often changed by fixes in this repository (count over 13 reference fixes): `src/requests/utils.py` (5), `src/requests/models.py` (3), `src/requests/adapters.py` (2), `src/requests/_types.py` (1), `src/requests/sessions.py` (1), `src/requests/compat.py` (1), `src/requests/exceptions.py` (1), `src/requests/status_codes.py` (1).
Tests most often extended by those fixes: `tests/test_requests.py`, `tests/test_utils.py`, `tests/test_adapters.py`.
Focused test run: `python -m pytest tests/test_<module>.py -q -x 2>&1 | tail -20` (the test file is usually named after the module).

Module map (top-level classes with their methods, public functions):
- `src/requests/__init__.py`: check_compatibility()
- `src/requests/_internal_utils.py`: to_native_string(), unicode_is_ascii()
- `src/requests/_types.py`: class SupportsRead(read), has_read(), class SupportsItems(items), is_prepared()
- `src/requests/adapters.py`: class BaseAdapter(__init__, send, close), class HTTPAdapter(__init__, init_poolmanager, proxy_manager_for, cert_verify, build_response, build_connection_pool_key_attributes, ...)
- `src/requests/api.py`: request(), get(), options(), head(), post(), put(), patch(), delete()
- `src/requests/auth.py`: class AuthBase(__call__), class HTTPBasicAuth(__init__, __init__, __init__, __call__), class HTTPProxyAuth(__call__), class HTTPDigestAuth(__init__, __init__, __init__, init_per_thread_state, build_digest_header, handle_redirect, ...)
- `src/requests/cookies.py`: class MockRequest(__init__, get_type, get_host, get_origin_req_host, get_full_url, is_unverifiable, ...), class MockResponse(__init__, info, getheaders), extract_cookies_to_jar(), get_cookie_header(), remove_cookie_by_name(), class CookieConflictError, class RequestsCookieJar(get, set, iterkeys, keys, itervalues, values, ...), create_cookie(), morsel_to_cookie(), cookiejar_from_dict(), cookiejar_from_dict(), cookiejar_from_dict(), merge_cookies()
- `src/requests/exceptions.py`: class RequestException(__init__), class InvalidJSONError, class JSONDecodeError(__init__), class HTTPError, class ConnectionError, class ProxyError, class SSLError, class Timeout, class ConnectTimeout, class ReadTimeout, class URLRequired, class TooManyRedirects, class MissingSchema, class InvalidSchema +11 more
- `src/requests/help.py`: info(), main()
- `src/requests/hooks.py`: default_hooks(), dispatch_hook()
- `src/requests/models.py`: class RequestEncodingMixin(path_url, _encode_params, _encode_params, _encode_params, _encode_params, _encode_params, ...), class RequestHooksMixin(register_hook, deregister_hook), class Request(__init__, prepare), class PreparedRequest(__init__, prepare, copy, prepare_method, _get_idna_encoded_host, prepare_url, ...), class Response(__init__, ok, is_redirect, is_permanent_redirect, next, apparent_encoding, ...)
- `src/requests/sessions.py`: merge_setting(), merge_hooks(), class SessionRedirectMixin(send, get_redirect_target, should_strip_auth, resolve_redirects, rebuild_auth, rebuild_proxies, ...), class Session(__init__, prepare_request, request, get, options, head, ...), session()
- `src/requests/structures.py`: class CaseInsensitiveDict(__init__, lower_items, copy), class LookupDict(__init__, get, get, get)
- `src/requests/utils.py`: dict_to_sequence(), super_len(), get_netrc_auth(), guess_filename(), extract_zipped_paths(), atomic_open(), from_key_val_list(), to_key_val_list(), to_key_val_list(), to_key_val_list(), parse_list_header(), parse_dict_header(), unquote_header_value(), dict_from_cookiejar() +28 more
