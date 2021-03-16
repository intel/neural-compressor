# -*- coding: utf-8 -*-
# Copyright (c) 2021 Intel Corporation
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#   http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Main endpoint for GUI."""
import os
from functools import wraps
from threading import Thread
from typing import Any, Callable

from flask import Flask
from flask import Request as WebRequest
from flask import Response as WebResponse
from flask import jsonify, request, send_file
from flask_cors import CORS
from flask_socketio import SocketIO

from lpot.ux.utils.exceptions import (
    AccessDeniedException,
    ClientErrorException,
    NotFoundException,
)
from lpot.ux.utils.utils import is_development_env, verify_file_path
from lpot.ux.web.communication import MessageQueue, Request
from lpot.ux.web.configuration import Configuration
from lpot.ux.web.router import Router

app = Flask(__name__, static_url_path="")
socketio = SocketIO()
router = Router()

METHODS = ["GET", "POST"]

# Suppress TensorFlow messages
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"


def run_server(configuration: Configuration) -> None:
    """Run webserver on specified scheme, address and port."""
    addr = configuration.ip
    port = configuration.port
    token = configuration.token

    cors_allowed_origins = f"{configuration.scheme}://{addr}:{port}"
    if is_development_env():
        cors_allowed_origins = "*"

    app.secret_key = token
    CORS(app, origins=cors_allowed_origins)
    socketio.init_app(
        app,
        cors_allowed_origins=cors_allowed_origins,
        max_http_buffer_size=2000,
    )

    args = {}
    if configuration.is_tls_used():
        args["certfile"] = configuration.tls_certificate
        args["keyfile"] = configuration.tls_key

    socketio.run(app, host=addr, port=port, **args)


@app.after_request
def block_iframe(response: WebResponse) -> WebResponse:
    """Block iframe and set others CSP."""
    response.headers["X-Frame-Options"] = "DENY"
    response.headers[
        "Content-Security-Policy"
    ] = "frame-ancestors 'none'; font-src 'self'; img-src 'self'; script-src 'self'"
    response.headers["Access-Control-Max-Age"] = "-1"
    return response


@app.after_request
def block_sniffing(response: WebResponse) -> WebResponse:
    """Block MIME sniffing."""
    response.headers["X-Content-Type-Options"] = "nosniff"
    return response


def require_api_token(func: Callable) -> Any:
    """Validate authorization token."""

    @wraps(func)
    def check_token(*args: str, **kwargs: str) -> Any:
        """Validate that correct token was provided."""
        provided_token = request.headers.get(
            "Authorization",
            request.args.to_dict().get("token", None),
        )

        if not app.secret_key == provided_token:
            return (
                "Invalid token, please use the URL displayed by the server on startup",
                403,
            )

        return func(*args, **kwargs)

    return check_token


@app.route("/", methods=METHODS)
def root() -> Any:
    """Serve JS application index."""
    return app.send_static_file("index.html")


@app.route("/file/<path:path>", methods=METHODS)
@require_api_token
def serve_from_filesystem(path: str) -> Any:
    """Serve any file from filesystem."""
    try:
        absolute_path = f"/{path}"
        verify_file_path(absolute_path)
        return send_file(absolute_path, as_attachment=True, cache_timeout=0)
    except NotFoundException as err:
        return str(err), 404
    except AccessDeniedException as err:
        return str(err), 403


@app.route("/api/<path:subpath>", methods=METHODS)
@require_api_token
def handle_api_call(subpath: str) -> Any:
    """Handle API access."""
    try:
        parameters = build_parameters(subpath, request)
        response = router.handle(parameters)
        return jsonify(response.data)
    except ClientErrorException as err:
        return str(err), 400
    except AccessDeniedException as err:
        return str(err), 403
    except NotFoundException as err:
        return str(err), 404


@app.route("/api/<path:subpath>", methods=["OPTIONS"])
def allow_api_call(subpath: str) -> Any:
    """Allow for API access."""
    return "OK"


@app.errorhandler(404)
def page_not_found(e: Any) -> Any:
    """Serve JS application index when no static file found."""
    return app.send_static_file("index.html")


@app.after_request
def disable_cache(response: WebResponse) -> WebResponse:
    """Disable cache on all requests."""
    response.headers["Cache-Control"] = "no-cache, no-store, must-revalidate"
    response.headers["Pragma"] = "no-cache"
    response.headers["Expires"] = "0"
    response.headers["Cache-Control"] = "public, max-age=0"
    return response


def build_parameters(endpoint: str, request: WebRequest) -> Request:
    """Build domain object from flask request."""
    data = request.get_json() if request.is_json else request.args.to_dict(flat=False)
    return Request(request.method, endpoint, data)


def web_socket_publisher(web_socket: SocketIO) -> None:
    """Send messages from queue via web-socket to GUI."""
    queue = MessageQueue()
    while True:
        message = queue.get()
        web_socket.emit(
            message.subject,
            {"status": message.status, "data": message.data},
            broadcast=True,
        )


publisher = Thread(
    target=web_socket_publisher,
    args=(socketio,),
)
publisher.daemon = True
publisher.start()
