---
name: repo-httpx
description: Map of the encode/httpx repository (the HTTPX HTTP client (sync/async clients, transports, models, URLs)). Load it first when the task's repository is encode/httpx; it says which file holds each class or function so you can go straight to read_file.
---

Repository `encode/httpx`: the HTTPX HTTP client (sync/async clients, transports, models, URLs). Source package: `httpx/`. Tests: `tests/` (14 `test_*.py` files at the top level).

Use this map to pick the file, then confirm with one scoped `git grep -n "<symbol>" httpx/` and `read_file` on that file only.
Do not grep the whole repository, docs or examples. The map was taken from one snapshot: when a name is missing, the code moved; fall back to `git grep`.

Files most often changed by fixes in this repository (count over 1 reference fixes): `src/ahttpx/_parsers.py` (1), `src/ahttpx/_pool.py` (1), `src/ahttpx/_server.py` (1), `src/httpx/_network.py` (1), `src/httpx/_parsers.py` (1), `src/httpx/_pool.py` (1), `src/httpx/_server.py` (1).
Tests most often extended by those fixes: `tests/test_parsers.py`.
Focused test run: `python -m pytest tests/test_<module>.py -q -x 2>&1 | tail -20` (the test file is usually named after the module).

Module map (top-level classes with their methods, public functions):
- `httpx/_api.py`: request(), stream(), get(), options(), head(), post(), put(), patch(), delete()
- `httpx/_auth.py`: class Auth(auth_flow, sync_auth_flow, async_auth_flow), class FunctionAuth(__init__, auth_flow), class BasicAuth(__init__, auth_flow, _build_auth_header), class NetRCAuth(__init__, auth_flow, _build_auth_header), class DigestAuth(__init__, auth_flow, _parse_challenge, _build_auth_header, _get_client_nonce, _get_header_value, ...), class _DigestAuthChallenge
- `httpx/_client.py`: class UseClientDefault, class ClientState, class BoundSyncStream(__init__, close), class BoundAsyncStream(__init__, aclose), class BaseClient(__init__, is_closed, trust_env, _enforce_trailing_slash, _get_proxy_map, timeout, ...), class Client(__init__, _init_transport, _init_proxy_transport, _transport_for_url, request, stream, ...), class AsyncClient(__init__, _init_transport, _init_proxy_transport, _transport_for_url, request, stream, ...)
- `httpx/_config.py`: class UnsetType, create_ssl_context(), class Timeout(__init__, as_dict), class Limits(__init__), class Proxy(__init__, raw_auth)
- `httpx/_content.py`: class ByteStream(__init__), class IteratorByteStream(__init__), class AsyncIteratorByteStream(__init__), class UnattachedStream, encode_content(), encode_urlencoded_data(), encode_multipart_data(), encode_text(), encode_html(), encode_json(), encode_request(), encode_response()
- `httpx/_decoders.py`: class ContentDecoder(decode, flush), class IdentityDecoder(decode, flush), class DeflateDecoder(__init__, decode, flush), class GZipDecoder(__init__, decode, flush), class BrotliDecoder(__init__, decode, flush), class ZStandardDecoder(__init__, decode, flush), class MultiDecoder(__init__, decode, flush), class ByteChunker(__init__, decode, flush), class TextChunker(__init__, decode, flush), class TextDecoder(__init__, decode, flush), class LineDecoder(__init__, decode, flush)
- `httpx/_exceptions.py`: class HTTPError(__init__, request, request), class RequestError(__init__), class TransportError, class TimeoutException, class ConnectTimeout, class ReadTimeout, class WriteTimeout, class PoolTimeout, class NetworkError, class ReadError, class WriteError, class ConnectError, class CloseError, class ProxyError +15 more
- `httpx/_main.py`: print_help(), get_lexer_for_response(), format_request_headers(), format_response_headers(), print_request_headers(), print_response_headers(), print_response(), format_certificate(), trace(), download_response(), validate_json(), validate_auth(), handle_help(), main()
- `httpx/_models.py`: class Headers(__init__, encoding, encoding, raw, keys, values, ...), class Request(__init__, _prepare, content, read, aread), class Response(__init__, _prepare, elapsed, elapsed, request, request, ...), class Cookies(__init__, extract_cookies, set_cookie_header, set, get, delete, ...)
- `httpx/_multipart.py`: get_multipart_boundary_from_content_type(), class DataField(__init__, render_headers, render_data, get_length, render), class FileField(__init__, get_length, render_headers, render_data, render), class MultipartStream(__init__, _iter_fields, iter_chunks, get_content_length, get_headers)
- `httpx/_status_codes.py`: class codes(get_reason_phrase, is_informational, is_success, is_redirect, is_client_error, is_server_error, ...)
- `httpx/_types.py`: class SyncByteStream(close), class AsyncByteStream(aclose)
- `httpx/_urlparse.py`: class ParseResult(authority, netloc, copy_with), urlparse(), encode_host(), normalize_port(), validate_path(), normalize_path(), PERCENT(), percent_encoded(), quote()
- `httpx/_urls.py`: class URL(__init__, scheme, raw_scheme, userinfo, username, password, ...), class QueryParams(__init__, keys, values, items, multi_items, get, ...)
- `httpx/_utils.py`: primitive_value_to_str(), get_environment_proxies(), to_bytes(), to_str(), to_bytes_or_str(), unquote(), peek_filelike_length(), class URLPattern(__init__, matches, priority), is_ipv4_hostname(), is_ipv6_hostname()
- `httpx/_transports/asgi.py`: is_running_trio(), create_event(), class ASGIResponseStream(__init__), class ASGITransport(__init__, handle_async_request)
- `httpx/_transports/base.py`: class BaseTransport(handle_request, close), class AsyncBaseTransport(handle_async_request, aclose)
- `httpx/_transports/default.py`: map_httpcore_exceptions(), class ResponseStream(__init__, close), class HTTPTransport(__init__, handle_request, close), class AsyncResponseStream(__init__, aclose), class AsyncHTTPTransport(__init__, handle_async_request, aclose)
- `httpx/_transports/mock.py`: class MockTransport(__init__, handle_request, handle_async_request)
- `httpx/_transports/wsgi.py`: class WSGIByteStream(__init__, close), class WSGITransport(__init__, handle_request)
