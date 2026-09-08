"""Allow MCP stdio child processes to inherit network proxy settings."""

from mcp.client import stdio

_NETWORK_ENV_VARS = (
    "HTTP_PROXY",
    "HTTPS_PROXY",
    "ALL_PROXY",
    "NO_PROXY",
    "http_proxy",
    "https_proxy",
    "all_proxy",
    "no_proxy",
    "UV_CONSTRAINT",
    "UV_HTTP_TIMEOUT",
)

stdio.DEFAULT_INHERITED_ENV_VARS = tuple(dict.fromkeys((*stdio.DEFAULT_INHERITED_ENV_VARS, *_NETWORK_ENV_VARS)))
