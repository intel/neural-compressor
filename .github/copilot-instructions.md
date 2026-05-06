# GitHub Copilot Instructions

This file configures GitHub Copilot (code suggestions and PR reviews) to follow the coding
standards enforced by the project's pre-commit hooks (`.pre-commit-config.yaml`) and
tool configuration (`pyproject.toml`).

---

## Python formatting – black

- Use **black** style formatting for all Python files (excluding `examples/`,
  `neural_compressor/torch/algorithms/fp8_quant/`, and `test/torch/`).
- Line length: **120 characters**.
- Use double quotes for strings.
- Indent with 4 spaces; never use tabs.
- Respect magic trailing commas.

## Import ordering – isort

- Sort imports with **isort** using the `black` profile.
- Maximum line length for imports: **120 characters**.
- Treat `neural_compressor` as a first-party package.
- Do not modify `__init__.py` files' imports.
- Applies to all Python files except `examples/`,
  `neural_compressor/torch/algorithms/fp8_quant/`, and `test/torch/`.

## Linting – ruff

- Follow **ruff** rules `E4`, `E7`, `E9`, and `F` (Pyflakes + pycodestyle errors).
- Target Python **3.10** syntax.
- Line length: **120 characters**; indent width: **4 spaces**.
- The following rule IDs are intentionally ignored in this project and must not be
  flagged as issues:
  - `E402` – module-level import not at top of file
  - `E501` – line too long
  - `E721` – type comparison (using `isinstance()` is preferred but not enforced)
  - `E722` – bare `except`
  - `E731` – lambda assigned to variable
  - `E741` – ambiguous variable names (`l`, `O`, `I`)
  - `F401` – imported but unused
  - `F403` / `F405` – star imports
  - `F841` – local variable assigned but never used
- Underscore-prefixed variables are allowed to be unused.
- Applies to all Python files except `examples/`,
  `neural_compressor/torch/algorithms/fp8_quant/`, and `test/torch/`.

## Docstrings – docformatter

- Format docstrings using **Google style**.
- Do not wrap summary lines or description lines (wrapping is disabled).
- Docstrings must be **black**-compatible.
- Applies to all Python files except `examples/`,
  `neural_compressor/torch/algorithms/fp8_quant/`, and `test/torch/`.

## Docstring examples in documentation – blacken-docs

- Code blocks inside Markdown and reStructuredText documentation files must also
  follow **black** formatting with line length **120**.
- Applies to all docs except `examples/`, `docs/source/JAX.md`,
  `neural_compressor/torch/algorithms/fp8_quant/`, and `test/torch/`.

## Security – bandit

- All code under `neural_compressor/` must pass **bandit** at low severity (`-lll`)
  and low confidence (`-iii`) thresholds.
- Do not suggest patterns that introduce common security issues such as:
  - Use of `subprocess` with `shell=True`
  - Hard-coded passwords or secrets
  - Use of `pickle`, `yaml.load` (without `Loader`), `eval`, or `exec`
  - Insecure use of temporary files or random number generators in security contexts

## Spelling – codespell

- Ensure all identifiers, comments, and documentation are correctly spelled.
- The project uses **codespell** to detect and auto-fix typos.

## Unused `# noqa` comments – yesqa

- Remove `# noqa` comments that are no longer suppressing any active lint warning.

## Debug statements

- Do not include `breakpoint()`, `pdb.set_trace()`, or any other debug statement in
  committed code.

## Whitespace and file hygiene

- No trailing whitespace in `.py`, `.rst`, `.cmake`, `.yaml`, or `.yml` files
  (except files under `examples/`, `neural_compressor/torch/utils/`,
  `neural_compressor/torch/algorithms/fp8_quant/`, and `test/torch/quantization/`).
- All `.py`, `.md`, `.rst`, `.yaml`, and `.yml` files must end with a single newline
  character (except files under `examples/`,
  `neural_compressor/torch/algorithms/fp8_quant/`, and
  `test/torch/algorithms/fp8_quant/`).
- JSON files must be valid (except `.vscode/settings_recommended.json`).
- YAML files must be valid.

## License headers

- Every new file under `neural_compressor/` with extension `.py`, `.yaml`, `.yml`,
  or `.sh` **must** include the Apache 2.0 license header at the top (within the
  first 40 lines). The header template is:

  ```
  # Copyright (c) YYYY Intel Corporation
  #
  # Licensed under the Apache License, Version 2.0 (the "License");
  # you may not use this file except in compliance with the License.
  # You may obtain a copy of the License at
  #
  #    http://www.apache.org/licenses/LICENSE-2.0
  #
  # Unless required by applicable law or agreed to in writing, software
  # distributed under the License is distributed on an "AS IS" BASIS,
  # WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
  # See the License for the specific language governing permissions and
  # limitations under the License.
  ```

- Do not add the header again if a `Copyright` notice already exists in the file.

## Requirements and dependency files

- Keep `requirements*.txt` files sorted and deduplicated (except those under
  `examples/`).

## Scope of these rules

These rules apply to all code suggestions and PR review comments made by Copilot in
this repository. When suggesting changes, Copilot should ensure the resulting code
would pass all the pre-commit hooks listed above. If a file path matches one of the
exclusion patterns for a given hook, that specific rule does not need to be applied
to that file.
