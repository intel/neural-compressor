# Copyright (c) 2022 Intel Corporation
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


import json

import pkg_resources
from jupyter_server.base.handlers import APIHandler
from jupyter_server.serverapp import ServerWebApplication
from jupyter_server.utils import url_path_join
from pathlib import Path


HERE = Path(__file__).parent.parent.resolve()
TMP_FILE = "tmp.py"

def setup_handlers(web_app: ServerWebApplication) -> None:
    host_pattern = ".*$"
    web_app.add_handlers(
        host_pattern,
        [
            (
                url_path_join(
                    web_app.settings["base_url"],
                    "/neural-compressor-ext-lab/optimize",
                ),
                OptimizeAPIHandler,
            )
        ],
    )
    web_app.add_handlers(
        host_pattern,
        [
            (
                url_path_join(
                    web_app.settings["base_url"], "/neural-compressor-ext-lab/version"
                ),
                VersionAPIHandler,
            )
        ],
    )


def check_plugin_version(handler: APIHandler):
    server_extension_version = pkg_resources.get_distribution(
        "neural-compressor-ext-lab"
    ).version
    lab_extension_version = handler.request.headers.get("Plugin-Version")
    version_matches = server_extension_version == lab_extension_version
    if not version_matches:
        handler.set_status(
            422,
            f"Mismatched versions of server extension ({server_extension_version}) "
            f"and lab extension ({lab_extension_version}). "
            f"Please ensure they are the same.",
        )
        handler.finish()
    return version_matches


class OptimizeAPIHandler(APIHandler):
    def post(self) -> None:
        if self.get_query_argument(
            "bypassVersionCheck", default=None
        ) is not None or check_plugin_version(self):
            data = json.loads(self.request.body.decode("utf-8"))
            print("Handle optimize request")
            notebook = data["notebook"]
            options = data.get("options", {})
            optimized_code = []
            with open( HERE/TMP_FILE, 'w+' ) as f:
                for code in data["code"]:
                    f.write("# this is the beginning of a single code snippet\n")
                    code_list = code.split("\n")
                    for line in code_list:
                        f.write(line+"\n")

            from neural_coder import enable
            enable(code=str(HERE/TMP_FILE), features=[data['formatter']], overwrite=True)
            
            with open( HERE/TMP_FILE, 'r' ) as f:
                content = f.read()
            optimized_code = content.split("# this is the beginning of a single code snippet\n")[1:]
            self.finish(json.dumps({"code": optimized_code}))

class VersionAPIHandler(APIHandler):
    def get(self) -> None:
        """Show what version is this server plguin on."""
        self.finish(
            json.dumps(
                {
                    "version": pkg_resources.get_distribution(
                        "neural-compressor-ext-lab"
                    ).version
                }
            )
        )
