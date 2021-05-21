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

"""Response generator."""
from typing import Optional

from flask import send_file
from werkzeug.wrappers import BaseResponse as Response

from lpot.ux.utils.exceptions import (
    AccessDeniedException,
    ClientErrorException,
    InternalException,
    NotFoundException,
)
from lpot.ux.utils.utils import verify_file_path


class ResponseGenerator:
    """Response generator class."""

    @staticmethod
    def serve_from_filesystem(
        path: str,
        mimetype: Optional[str] = None,
    ) -> Response:
        """Serve any file from filesystem."""
        verify_file_path(path)
        return send_file(
            path,
            mimetype=mimetype,
            as_attachment=False,
            cache_timeout=0,
        )

    @staticmethod
    def add_refresh(response: Response, refresh_time: int) -> Response:
        """Add Refresh header to response."""
        response.headers["refresh"] = refresh_time
        return response

    @staticmethod
    def from_exception(exception: Exception) -> Response:
        """Create Response from Exception."""
        return Response(
            response=str(exception),
            status=ResponseGenerator.get_status_code_for_exception(exception),
        )

    @staticmethod
    def get_status_code_for_exception(exception: Exception) -> int:
        """Get HTTP status code for Exception."""
        if isinstance(exception, ClientErrorException):
            return 400
        if isinstance(exception, AccessDeniedException):
            return 403
        if isinstance(exception, NotFoundException):
            return 404
        if isinstance(exception, InternalException):
            return 500
        return 500
