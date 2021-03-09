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
"""Download model from Examples."""

import os
import tarfile
import zipfile
from typing import Any, Dict, Optional, Tuple

import requests

from lpot.ux.utils.consts import github_info
from lpot.ux.utils.exceptions import ClientErrorException
from lpot.ux.utils.logger import log
from lpot.ux.utils.utils import is_development_env, load_model_config
from lpot.ux.web.communication import MessageQueue


class Downloader:
    """UX model downloader class."""

    def __init__(self, data: Dict[str, Any]) -> None:
        """Initialize Downloader class."""
        self.request_id: str = str(data.get("id", ""))
        self.framework: str = data.get("framework", "")
        self.domain: str = data.get("domain", "")
        self.model: str = data.get("model", "")
        self.workspace_path: str = data.get("workspace_path", "")
        self.progress_steps: Optional[int] = data.get("progress_steps", None)
        self.download_dir: str = ""
        self.mq = MessageQueue()

    def download_config(self) -> None:
        """Find yaml config resource and initialize downloading."""
        if not (
            self.request_id
            and self.framework
            and self.domain
            and self.model
            and self.workspace_path
        ):
            message = "Missing request id, workspace path, framework, domain or model."
            self.mq.post_error(
                "download_finish",
                {"message": message, "code": 404, "id": self.request_id},
            )
            raise ClientErrorException(message)

        model_config = load_model_config()
        model_info = (
            model_config.get(self.framework, {})
            .get(self.domain, {})
            .get(self.model, None)
        )

        if model_info is None:
            raise Exception(
                f"{self.framework} {self.domain} {self.model} is not supported.",
            )

        self.download_dir = os.path.join(
            self.workspace_path,
            "examples",
            self.framework,
            self.domain,
            self.model,
        )

        self.download_yaml_config(model_info)

    def download_yaml_config(self, model_info: Dict[str, Any]) -> None:
        """Download config from GitHub for specified model."""
        yaml_relative_location = model_info.get("yaml", "")
        if not yaml_relative_location:
            message = "Missing yaml location."
            self.mq.post_error(
                "download_finish",
                {"message": message, "code": 404, "id": self.request_id},
            )
            raise ClientErrorException(message)

        url, headers = self.get_yaml_url(
            yaml_relative_location,
        )

        download_path = os.path.join(
            self.download_dir,
            os.path.basename(yaml_relative_location),
        )
        self.download_file(
            url=url,
            headers=headers,
            download_path=download_path,
        )

        self.mq.post_success(
            "download_finish",
            {
                "id": self.request_id,
                "path": download_path,
            },
        )

    def download_model(self) -> None:
        """Find model resource and initialize downloading."""
        model_config = load_model_config()
        model_info = (
            model_config.get(self.framework, {})
            .get(self.domain, {})
            .get(self.model, None)
        )

        if model_info is None:
            raise Exception(
                f"{self.framework} {self.domain} {self.model} is not supported.",
            )

        self.download_dir = os.path.join(
            self.workspace_path,
            "examples",
            self.framework,
            self.domain,
            self.model,
        )

        self.download(model_info)

    def download(self, model_info: Dict[str, Any]) -> None:
        """Download specified model."""
        download_info = model_info.get("download", None)
        if download_info is None:
            message = "Model download is not supported."
            self.mq.post_error(
                "download_finish",
                {"message": message, "code": 404, "id": self.request_id},
            )
            raise ClientErrorException(message)

        url = download_info.get("url")
        filename = download_info.get("filename")
        is_archived = download_info.get("is_archived")
        if not (url and filename):
            message = "Could not found download link for model or output file name."
            self.mq.post_error(
                "download_finish",
                {"message": message, "code": 404, "id": self.request_id},
            )
            raise ClientErrorException(message)

        download_path = os.path.join(self.download_dir, filename)
        if is_archived:
            download_path = os.path.join(self.download_dir, url.split("/")[-1])

        self.download_file(
            url=url,
            download_path=download_path,
        )

        model_path = download_path
        if is_archived:
            model_path = self.unpack_archive(download_path, filename)

        self.mq.post_success(
            "download_finish",
            {
                "id": self.request_id,
                "path": model_path,
            },
        )

    def download_file(
        self,
        url: str,
        download_path: str,
        headers: Optional[dict] = {},
    ) -> None:
        """Download specified file."""
        try:
            with requests.get(
                url,
                allow_redirects=True,
                stream=True,
                headers=headers,
            ) as r:
                r.raise_for_status()
                os.makedirs(os.path.dirname(download_path), exist_ok=True)
                with open(download_path, "wb") as f:
                    log.debug(f"Download file from {url} to {download_path}")
                    total_length = r.headers.get("content-length")
                    self.mq.post_success(
                        "download_start",
                        {
                            "message": "started",
                            "id": self.request_id,
                            "url": url,
                        },
                    )
                    if total_length is None:
                        f.write(r.content)
                        return
                    downloaded = 0
                    last_progress = 0
                    total_size = int(total_length)
                    for data in r.iter_content(chunk_size=4096):
                        downloaded += len(data)
                        f.write(data)
                        if self.progress_steps:
                            progress = int(100 * downloaded / total_size)
                            if (
                                last_progress != progress
                                and progress % int(100 / self.progress_steps) == 0
                            ):
                                self.mq.post_success(
                                    "download_progress",
                                    {
                                        "id": self.request_id,
                                        "progress": f"{downloaded}/{total_size}",
                                    },
                                )
                                log.debug(f"Download progress: {progress}%")
                                last_progress = progress
        except requests.exceptions.HTTPError:
            message = f"Error downloading file from {url} to {download_path}"
            self.mq.post_error(
                "download_finish",
                {
                    "message": message,
                    "code": 404,
                    "id": self.request_id,
                },
            )
            return

    def unpack_archive(self, archive_path: str, filename: str) -> str:
        """Unpack archive and return path to unpacked model."""
        self.mq.post_success(
            "unpack_start",
            {
                "id": self.request_id,
            },
        )
        log.debug(f"Unpacking {archive_path}")

        if zipfile.is_zipfile(archive_path):
            z = zipfile.ZipFile(archive_path)
            z.extractall(self.download_dir)

        elif tarfile.is_tarfile(archive_path):
            t = tarfile.open(archive_path, "r:gz")
            t.extractall(self.download_dir)

        else:
            message = (
                "Could unpack an archive. Supported archive types are zip and tar.gz."
            )
            self.mq.post_error(
                "unpack_finish",
                {"message": message, "code": 404, "id": self.request_id},
            )
            raise ClientErrorException(message)

        os.remove(archive_path)
        unpacked_path = os.path.join(self.download_dir, filename)
        self.mq.post_success(
            "unpack_finish",
            {"id": self.request_id, "path": unpacked_path},
        )
        log.debug(f"Model file has been extracted to {unpacked_path}")
        return unpacked_path

    def get_yaml_url(
        self,
        yaml_relative_location: str,
    ) -> Tuple[str, dict]:
        """Get url for yaml config download."""
        if is_development_env():
            from urllib.parse import quote_plus

            file_path = quote_plus(
                os.path.join(
                    "examples",
                    self.framework,
                    self.domain,
                    yaml_relative_location,
                ),
            )
            url = os.path.join(
                os.environ["LPOT_PROJECT_URL"],
                file_path,
                "raw?ref=developer",
            )
            headers = {"Private-Token": os.environ.get("LPOT_TOKEN")}
            return url, headers
        user = github_info.get("user")
        repository = github_info.get("repository")
        tag = github_info.get("tag")

        if not (user, repository, tag):
            message = "Missing github repository information."
            self.mq.post_error(
                "download_finish",
                {"message": message, "code": 500, "id": self.request_id},
            )
            raise ClientErrorException(message)
        url_prefix = f"https://raw.githubusercontent.com/{user}/{repository}/{tag}/"
        url = os.path.join(
            url_prefix,
            "examples",
            self.framework,
            self.domain,
            yaml_relative_location,
        )
        return url, {}
