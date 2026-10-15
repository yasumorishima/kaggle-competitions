---
name: repo-fastapi
description: Map of the fastapi/fastapi repository (the FastAPI web framework (routing, dependency injection, OpenAPI generation, request validation on Pydantic)). Load it first when the task's repository is fastapi/fastapi; it says which file holds each class or function so you can go straight to read_file.
---

Repository `fastapi/fastapi`: the FastAPI web framework (routing, dependency injection, OpenAPI generation, request validation on Pydantic). Source package: `fastapi/`. Tests: `tests/` (210 `test_*.py` files at the top level).

Use this map to pick the file, then confirm with one scoped `git grep -n "<symbol>" fastapi/` and `read_file` on that file only.
Do not grep the whole repository, docs or examples. The map was taken from one snapshot: when a name is missing, the code moved; fall back to `git grep`.

Files most often changed by fixes in this repository (count over 67 reference fixes): `fastapi/dependencies/utils.py` (27), `fastapi/routing.py` (16), `fastapi/_compat/v2.py` (13), `fastapi/openapi/utils.py` (12), `fastapi/applications.py` (7), `fastapi/dependencies/models.py` (7), `fastapi/params.py` (5), `fastapi/exceptions.py` (4), `fastapi/temp_pydantic_v1_params.py` (3), `fastapi/utils.py` (3), `fastapi/sse.py` (2), `fastapi/responses.py` (2).
Tests most often extended by those fixes: `tests/test_compat.py`, `tests/test_get_model_definitions_formfeed_escape.py`, `tests/test_compat_params_v1.py`, `tests/test_router_include_context.py`, `tests/test_sse.py`, `tests/test_query_cookie_header_model_extra_params.py`.
Focused test run: `python -m pytest tests/test_<module>.py -q -x 2>&1 | tail -20` (the test file is usually named after the module).

Module map (top-level classes with their methods, public functions):
- `fastapi/applications.py`: class FastAPI(__init__, build_middleware_stack, openapi, setup, __call__, add_api_route, ...)
- `fastapi/background.py`: class BackgroundTasks(add_task, __call__)
- `fastapi/cli.py`: main()
- `fastapi/concurrency.py`: contextmanager_in_threadpool()
- `fastapi/datastructures.py`: class UploadFile(write, read, seek, close, _validate), class DefaultPlaceholder(__init__), Default()
- `fastapi/encoders.py`: isoformat(), decimal_encoder(), generate_encoders_by_class_tuples(), jsonable_encoder()
- `fastapi/exception_handlers.py`: http_exception_handler(), request_validation_exception_handler(), websocket_request_validation_exception_handler()
- `fastapi/exceptions.py`: class EndpointContext, class HTTPException(__init__), class WebSocketException(__init__), class FastAPIError, class DependencyScopeError, class ValidationException(__init__, errors, _format_endpoint_context), class RequestValidationError(__init__), class WebSocketRequestValidationError(__init__), class ResponseValidationError(__init__), class PydanticV1NotSupportedError, class FastAPIDeprecationWarning
- `fastapi/param_functions.py`: Path(), Query(), Header(), Cookie(), Body(), Form(), File(), Depends(), Security()
- `fastapi/params.py`: class ParamTypes, class Param(__init__), class Path(__init__), class Query(__init__), class Header(__init__), class Cookie(__init__), class Body(__init__), class Form(__init__), class File(__init__), class Depends, class Security
- `fastapi/responses.py`: class _UjsonModule(dumps), class _OrjsonModule(dumps), class UJSONResponse(render), class ORJSONResponse(render)
- `fastapi/routing.py`: request_response(), websocket_session(), class _AsyncLiftContextManager(__init__), class _DefaultLifespan(__init__, __call__), serialize_response(), run_endpoint_function(), get_request_handler(), get_websocket_app(), class APIWebSocketRoute(__init__, matches), class _RouteWithPath, class _APIRouteLike, class APIRoute(__init__, get_route_handler, matches, handle), class _RouterIncludeContext(for_include, combine, path_for), class _EffectiveRouteContext(from_api_route, from_frontend_route_group, matches, url_path_for) +7 more
- `fastapi/sse.py`: class EventSourceResponse, class ServerSentEvent(_check_data_exclusive), format_sse_event()
- `fastapi/utils.py`: is_body_allowed_for_status_code(), get_path_param_names(), create_model_field(), generate_operation_id_for_path(), generate_unique_id(), deep_dict_update(), get_value_or_default()
- `fastapi/_compat/shared.py`: lenient_issubclass(), field_annotation_is_sequence(), value_is_sequence(), field_annotation_is_complex(), field_annotation_is_scalar(), field_annotation_is_scalar_sequence(), is_bytes_or_nonable_bytes_annotation(), is_uploadfile_or_nonable_uploadfile_annotation(), is_bytes_sequence_annotation(), is_uploadfile_sequence_annotation(), is_pydantic_v1_model_instance(), is_pydantic_v1_model_class(), annotation_is_pydantic_v1()
- `fastapi/_compat/v2.py`: evaluate_forwardref(), class GenerateJsonSchema(bytes_schema), asdict(), class ModelField(alias, validation_alias, serialization_alias, default, get_default, validate, ...), get_schema_from_model_field(), get_definitions(), is_scalar_field(), copy_field_info(), serialize_sequence_value(), get_missing_field_error(), create_body_model(), get_model_fields(), get_cached_model_fields(), normalize_name() +5 more
- `fastapi/dependencies/models.py`: class Dependant, class _CallIdentity(__init__)
- `fastapi/dependencies/utils.py`: ensure_multipart_is_installed(), get_parameterless_sub_dependant(), get_flat_params(), get_typed_signature(), get_typed_annotation(), get_typed_return_annotation(), get_stream_item_type(), get_dependant(), add_non_field_param_to_dependency(), class ParamDetails, analyze_param(), add_param_to_fields(), class SolvedDependency, solve_dependencies() +4 more
- `fastapi/middleware/asyncexitstack.py`: class AsyncExitStackMiddleware(__init__, __call__)
- `fastapi/openapi/docs.py`: get_swagger_ui_html(), get_redoc_html(), get_swagger_ui_oauth2_redirect_html()
- `fastapi/openapi/models.py`: class BaseModelWithConfig, class Contact, class License, class Info, class ServerVariable, class Server, class Reference, class Discriminator, class XML, class ExternalDocumentation, class Schema, class Example, class ParameterInType, class Encoding +26 more
- `fastapi/openapi/utils.py`: class _OpenAPIDependencyData, get_openapi_operation_request_body(), generate_operation_id(), generate_operation_summary(), get_openapi_operation_metadata(), get_openapi_path(), get_fields_from_routes(), get_openapi()
- `fastapi/security/api_key.py`: class APIKeyBase(__init__, make_not_authenticated_error, check_api_key), class APIKeyQuery(__init__, __call__), class APIKeyHeader(__init__, __call__), class APIKeyCookie(__init__, __call__)
- `fastapi/security/base.py`: class SecurityBase
- `fastapi/security/http.py`: class HTTPBasicCredentials, class HTTPAuthorizationCredentials, class HTTPBase(__init__, make_authenticate_headers, make_not_authenticated_error, __call__), class HTTPBasic(__init__, make_authenticate_headers, __call__), class HTTPBearer(__init__, __call__), class HTTPDigest(__init__, __call__)
- `fastapi/security/oauth2.py`: class OAuth2PasswordRequestForm(__init__), class OAuth2PasswordRequestFormStrict(__init__), class OAuth2(__init__, make_not_authenticated_error, __call__), class OAuth2PasswordBearer(__init__, __call__), class OAuth2AuthorizationCodeBearer(__init__, __call__), class SecurityScopes(__init__)
- `fastapi/security/open_id_connect_url.py`: class OpenIdConnect(__init__, make_not_authenticated_error, __call__)
- `fastapi/security/utils.py`: get_authorization_scheme_param()
- `fastapi/telemetry/_api.py`: class TelemetryConfig, class TelemetryData, class _RequestTelemetry, get_telemetry_data()
- `fastapi/telemetry/_asgi.py`: class _HeadersGetter(get, keys), class ExceptionTelemetryMiddleware(__init__, __call__), class NativeTelemetry(__init__, enabled, _instruments, __call__)
- `fastapi/telemetry/_runtime.py`: lifespan()
