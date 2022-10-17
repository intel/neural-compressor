"use strict";
(self["webpackChunkneural_compressor_ext_lab"] = self["webpackChunkneural_compressor_ext_lab"] || []).push([["lib_index_js"],{

/***/ "./lib/constants.js":
/*!**************************!*\
  !*** ./lib/constants.js ***!
  \**************************/
/***/ ((__unused_webpack_module, __webpack_exports__, __webpack_require__) => {

__webpack_require__.r(__webpack_exports__);
/* harmony export */ __webpack_require__.d(__webpack_exports__, {
/* harmony export */   "Constants": () => (/* binding */ Constants)
/* harmony export */ });
var Constants;
(function (Constants) {
    Constants.SHORT_PLUGIN_NAME = 'neural_compressor_ext_lab';
    Constants.WORK_PATH = "neural_coder_workspace/";
    Constants.ICON_FORMAT_ALL_SVG = '<svg xmlns="http://www.w3.org/2000/svg" xmlns:xlink="http://www.w3.org/1999/xlink" style="margin: auto; background: rgb(255, 255, 255); display: block; shape-rendering: auto;" width="53px" height="53px" viewBox="0 0 100 100" preserveAspectRatio="xMidYMid"><circle cx="50" cy="50" r="32" stroke-width="8" stroke="#e15b64" stroke-dasharray="50.26548245743669 50.26548245743669" fill="none" stroke-linecap="round"><animateTransform attributeName="transform" type="rotate" dur="1s" repeatCount="indefinite" keyTimes="0;1" values="0 50 50;360 50 50"></animateTransform></circle><circle cx="50" cy="50" r="23" stroke-width="8" stroke="#f8b26a" stroke-dasharray="36.12831551628262 36.12831551628262" stroke-dashoffset="36.12831551628262" fill="none" stroke-linecap="round"><animateTransform attributeName="transform" type="rotate" dur="1s" repeatCount="indefinite" keyTimes="0;1" values="0 50 50;-360 50 50"></animateTransform></circle></svg>';
    Constants.ICON_RUN = '<svg xmlns="http://www.w3.org/2000/svg" xmlns:xlink="http://www.w3.org/1999/xlink" xmlns:svgjs="http://svgjs.com/svgjs" version="1.1" width="512" height="512" x="0" y="0" viewBox="0 0 24 24" style="enable-background:new 0 0 512 512" xml:space="preserve" class=""><g><g xmlns="http://www.w3.org/2000/svg" clip-rule="evenodd" fill="rgb(0,0,0)" fill-rule="evenodd"><path d="m3.09467 3.09467c1.33447-1.33447 3.33136-1.84467 5.90533-1.84467h6c2.574 0 4.5709.5102 5.9053 1.84467 1.3345 1.33447 1.8447 3.33136 1.8447 5.90533v6c0 2.574-.5102 4.5709-1.8447 5.9053-1.3344 1.3345-3.3313 1.8447-5.9053 1.8447h-6c-2.57397 0-4.57086-.5102-5.90533-1.8447-1.33447-1.3344-1.84467-3.3313-1.84467-5.9053v-2.05c0-.4142.33579-.75.75-.75s.75.3358.75.75v2.05c0 2.426.4898 3.9291 1.40533 4.8447.91553.9155 2.41864 1.4053 4.84467 1.4053h6c2.426 0 3.9291-.4898 4.8447-1.4053.9155-.9156 1.4053-2.4187 1.4053-4.8447v-6c0-2.42603-.4898-3.92914-1.4053-4.84467-.9156-.91553-2.4187-1.40533-4.8447-1.40533h-6c-2.42603 0-3.92914.4898-4.84467 1.40533s-1.40533 2.41864-1.40533 4.84467c0 .41421-.33579.75-.75.75s-.75-.33579-.75-.75c0-2.57397.5102-4.57086 1.84467-5.90533z" fill="#505361" data-original="#000000" class=""/><path d="m10.355 9.23276c-.2302.13229-.505.4923-.505 1.28724v2.96c0 .7885.2739 1.1502.5061 1.2841.2324.1342.6841.1907 1.3697-.2041.3589-.2066.8175-.0832 1.0242.2758.2066.3589.0832.8175-.2758 1.0242-.9644.5552-2.0127.6967-2.86779.2034-.85535-.4936-1.25641-1.4719-1.25641-2.5834v-2.96c0-1.11506.40022-2.09505 1.25754-2.58776.85596-.49195 1.90416-.34642 2.86666.20779l.0012.00067 2.5588 1.47933c-.0002-.00012.0002.00013 0 0 .9642.55537 1.6133 1.39217 1.6133 2.37997 0 .9881-.6487 1.8246-1.6133 2.38-.3589.2066-.8175.0832-1.0242-.2758-.2066-.3589-.0832-.8175.2758-1.0242.6854-.3946.8617-.8131.8617-1.08s-.1763-.6854-.8617-1.08l-2.56-1.48003c.0002.0001-.0002-.00011 0 0-.6871-.39546-1.1394-.34022-1.3708-.20721z" fill="#505361" data-original="#000000" class=""/></g></g></svg>';
    Constants.SVG = '<svg xmlns="http://www.w3.org/2000/svg" width="16" viewBox="0 0 18 18" data-icon="ui-components:caret-down-empty"><g xmlns="http://www.w3.org/2000/svg" class="jp-icon3" fill="#616161" shape-rendering="geometricPrecision"><path d="M5.2,5.9L9,9.7l3.8-3.8l1.2,1.2l-4.9,5l-4.9-5L5.2,5.9z"></path></g></svg>';
    Constants.LONG_PLUGIN_NAME = `@rya/${Constants.SHORT_PLUGIN_NAME}`;
    Constants.SETTINGS_SECTION = `${Constants.LONG_PLUGIN_NAME}:settings`;
    Constants.COMMAND_SECTION_NAME = 'Jupyterlab Code Optimizer';
    Constants.PLUGIN_VERSION = '0.1.0';
})(Constants || (Constants = {}));


/***/ }),

/***/ "./lib/deepcoder.js":
/*!**************************!*\
  !*** ./lib/deepcoder.js ***!
  \**************************/
/***/ ((__unused_webpack_module, __webpack_exports__, __webpack_require__) => {

__webpack_require__.r(__webpack_exports__);
/* harmony export */ __webpack_require__.d(__webpack_exports__, {
/* harmony export */   "JupyterlabNotebookCodeOptimizer": () => (/* binding */ JupyterlabNotebookCodeOptimizer)
/* harmony export */ });
/* harmony import */ var _jupyterlab_notebook__WEBPACK_IMPORTED_MODULE_0__ = __webpack_require__(/*! @jupyterlab/notebook */ "webpack/sharing/consume/default/@jupyterlab/notebook");
/* harmony import */ var _jupyterlab_notebook__WEBPACK_IMPORTED_MODULE_0___default = /*#__PURE__*/__webpack_require__.n(_jupyterlab_notebook__WEBPACK_IMPORTED_MODULE_0__);
/* harmony import */ var _utils__WEBPACK_IMPORTED_MODULE_1__ = __webpack_require__(/*! ./utils */ "./lib/utils.js");
/* harmony import */ var _constants__WEBPACK_IMPORTED_MODULE_2__ = __webpack_require__(/*! ./constants */ "./lib/constants.js");



class JupyterlabCodeOptimizer {
    constructor(panel) {
        this.working = false;
        this.panel = panel;
        this.tmp_path = "tmp.py";
        this.rand = _utils__WEBPACK_IMPORTED_MODULE_1__["default"].GetRandomNum(0, 200);
        this.log_path = _constants__WEBPACK_IMPORTED_MODULE_2__.Constants.WORK_PATH + "NeuralCoder" + this.rand + ".log";
        this.tmp_log_path = _constants__WEBPACK_IMPORTED_MODULE_2__.Constants.WORK_PATH + "NeuralCoder_tmp" + ".log";
        this.cells = [];
    }
    async optimizeCode(code, formatter, name, next, options, notebook, panel, cell, run) {
        let codes = [];
        code.forEach(function (value) {
            value = value.replace(/('\\n')/g, '^^^');
            value = value.replace(/\\n"/g, '###');
            value = value.replace(/\\n'/g, '###');
            value = value.replace(/"\\n/g, '@@');
            value = value.replace(/'\\n/g, '@@');
            value = value.replace(/\n/g, '\\n');
            value = value.replace(/"/g, '+++');
            value = value.replace(/,/g, '$');
            codes.push(value);
        });
        let gen_code = `code = "${codes}"\ncodes = code.split(',')\nwith open( '${this.tmp_path}', 'w+' ) as f:\n    for i in range(0,len(codes)):\n        f.write('# this is the beginning of a single code snippet\\n')\n        code_list = codes[i].replace('$',',').replace('+++','\"').split('\\n')\n        for line in code_list:\n            if('split(^^^)' in line):\n                    line=line.replace('split(^^^)', 'split(\\'\\\\n\\')')\n            if('###' in line):\n                    line=line.replace('###', '\\\\n\"')\n            if('@@' in line):\n                    line=line.replace('@@', '\"\\\\n')\n            f.write(line+'\\n')`;
        const expr = { code_list: `code_list` };
        _utils__WEBPACK_IMPORTED_MODULE_1__["default"].sendKernelRequestFromNotebook(panel, gen_code, expr, false);
        if (options === 'normal') {
            let runcode = `from neural_coder import enable\nenable(code="${this.tmp_path}",features=["${formatter}"], overwrite=True)`;
            let expr = { sum: ` ` };
            _utils__WEBPACK_IMPORTED_MODULE_1__["default"].sendKernelRequestFromNotebook(panel, runcode, expr, false);
            let run_code1 = `with open("${this.tmp_path}", 'r') as f:\n    optimized_code = f.read()\n`;
            let expr1 = { optimizedCode: "optimized_code" };
            let result2 = _utils__WEBPACK_IMPORTED_MODULE_1__["default"].sendKernelRequestFromNotebook(panel, run_code1, expr1, false);
            result2.then(value => {
                var _a, _b, _c, _d;
                let optimizedTexts = Object.values(value.optimizedCode.data)[0];
                let optimizeCodes = optimizedTexts.split('# this is the beginning of a single code snippet\\n').slice(1);
                optimizeCodes[optimizeCodes.length - 1] = optimizeCodes[optimizeCodes.length - 1].slice(0, -3);
                for (let i = 0; i < optimizeCodes.length; ++i) {
                    const cell = this.cells[i];
                    const currentTexts = this.cells.map(cell => cell.model.value.text);
                    const currentText = currentTexts[i];
                    let optimizedtext = optimizeCodes[i];
                    optimizedtext = optimizedtext.replace(/\\'\\\\n\\'/g, "^^^");
                    optimizedtext = optimizedtext.replace(/\\\\n"/g, "+++");
                    optimizedtext = optimizedtext.replace(/\\\\n'/g, "+++");
                    optimizedtext = optimizedtext.replace(/"\\\\n/g, "@@@");
                    optimizedtext = optimizedtext.replace(/'\\\\n/g, "@@@");
                    optimizedtext = optimizedtext.replace(/\\n/g, '\n');
                    optimizedtext = optimizedtext.replace(/\\'/g, "'");
                    optimizedtext = optimizedtext.replace(/\^\^\^/g, "'\\n'");
                    optimizedtext = optimizedtext.replace(/\+\+\+/g, "\\n\"");
                    optimizedtext = optimizedtext.replace(/\@\@\@/g, "\"\\n");
                    if (cell.model.value.text === currentText) {
                        cell.model.value.text = optimizedtext;
                    }
                    const run_svg = document.createElement("svg");
                    run_svg.innerHTML = _constants__WEBPACK_IMPORTED_MODULE_2__.Constants.ICON_RUN;
                    (_d = (_c = (_b = (_a = run === null || run === void 0 ? void 0 : run.node.firstChild) === null || _a === void 0 ? void 0 : _a.firstChild) === null || _b === void 0 ? void 0 : _b.firstChild) === null || _c === void 0 ? void 0 : _c.firstChild) === null || _d === void 0 ? void 0 : _d.replaceWith(run_svg);
                }
            });
        }
        else {
            if (formatter === '') {
                if (this.markdown) {
                    this.markdown.model.value.text += "[NeuralCoder INFO] Enabling and Benchmarking for The Original Model ......  \n";
                }
                // cell.outputArea.node.innerText += "[NeuralCoder INFO] Enabling and Benchmarking for The Original Model ......\n"
                let runcode1 = `with open("${this.log_path}", 'a' ) as f:\n       f.write("[NeuralCoder INFO] Enabling and Benchmarking for The Original Model ......\\n")`;
                let expr1 = { path: "" };
                _utils__WEBPACK_IMPORTED_MODULE_1__["default"].sendKernelRequestFromNotebook(panel, runcode1, expr1, false);
                let runcode = `from neural_coder import enable\nperfomance, mode, path = enable(code="${this.tmp_path}",features=[], run_bench=True, args="${options}")\nwith open(path + '/bench.log', 'r') as f:\n    logs = f.readlines()\nlog_line = logs[4]\nlog = log_line.split("[")[1].split("]")[0]`;
                let expr = { path: "path", log: "log" };
                let result = _utils__WEBPACK_IMPORTED_MODULE_1__["default"].sendKernelRequestFromNotebook(panel, runcode, expr, false);
                let fps;
                result.then(value => {
                    fps = Object.values(value.log.data)[0];
                    if (this.markdown) {
                        this.markdown.model.value.text += `[NeuralCoder INFO] Benchmark Result (Performance) of The Original Model is ${fps} (samples/second)  \n`;
                    }
                    // cell.outputArea.node.innerText += `[NeuralCoder INFO] Benchmark Result (Performance) of The Original Model is ${fps} (samples/second)\n`
                    let text = `[NeuralCoder INFO] Benchmark Result (Performance) of The Original Model is ${fps} (samples/second)\\n`;
                    let runcode = `with open("${this.log_path}", 'a' ) as f:\n   f.write("${text}")`;
                    let expr = { path: "" };
                    _utils__WEBPACK_IMPORTED_MODULE_1__["default"].sendKernelRequestFromNotebook(this.panel, runcode, expr, false);
                    if (this.markdown) {
                        this.markdown.model.value.text += `[NeuralCoder INFO] Enabling and Benchmarking for ${next} ......  \n`;
                    }
                    // cell.outputArea.node.innerText += `[NeuralCoder INFO] Enabling and Benchmarking for ${next} ......\n`
                    let runcode1 = `with open("${this.log_path}", 'a' ) as f:\n       f.write("[NeuralCoder INFO] Enabling and Benchmarking for ${next} ......\\n")`;
                    let expr1 = { path: "" };
                    _utils__WEBPACK_IMPORTED_MODULE_1__["default"].sendKernelRequestFromNotebook(panel, runcode1, expr1, false);
                    let runcode2 = `with open("${this.tmp_log_path}", 'a' ) as f:\n       f.write("${text}")`;
                    let expr2 = { path: "" };
                    _utils__WEBPACK_IMPORTED_MODULE_1__["default"].sendKernelRequestFromNotebook(this.panel, runcode2, expr2, false);
                });
            }
            else {
                let runcode = `from neural_coder import enable\nperfomance, mode, path = enable(code="${this.tmp_path}", features=["${formatter}"], run_bench=True, args="${options}")\nwith open(path + '/bench.log', 'r') as f:\n    logs = f.readlines()\nlog_line = logs[4]\nlog = log_line.split("[")[1].split("]")[0]`;
                let expr = { path: "path", log: "log" };
                let result = _utils__WEBPACK_IMPORTED_MODULE_1__["default"].sendKernelRequestFromNotebook(panel, runcode, expr, false);
                let fps;
                result.then(value => {
                    fps = Object.values(value.log.data)[0];
                    if (this.markdown) {
                        this.markdown.model.value.text += `[NeuralCoder INFO] Benchmark Result (Performance) of ${name} is ${fps} (samples/second)  \n`;
                    }
                    // cell.outputArea.node.innerText += `[NeuralCoder INFO] Benchmark Result (Performance) of ${name} is ${fps} (FPS)\n`
                    let text = `[NeuralCoder INFO] Benchmark Result (Performance) of ${name} is ${fps} (samples/second)\\n`;
                    let runcode = `with open("${this.log_path}", 'a' ) as f:\n       f.write("${text}")`;
                    let expr = { path: "" };
                    _utils__WEBPACK_IMPORTED_MODULE_1__["default"].sendKernelRequestFromNotebook(this.panel, runcode, expr, false);
                    if (next !== '') {
                        if (this.markdown) {
                            this.markdown.model.value.text += `[NeuralCoder INFO] Enabling and Benchmarking for ${next} ......  \n`;
                        }
                        // cell.outputArea.node.innerText += `[NeuralCoder INFO] Enabling and Benchmarking for ${next} ......\n`
                        let runcode2 = `with open("${this.log_path}", 'a' ) as f:\n       f.write("[NeuralCoder INFO] Enabling and Benchmarking for ${next} ......\\n")`;
                        let expr2 = { path: "" };
                        _utils__WEBPACK_IMPORTED_MODULE_1__["default"].sendKernelRequestFromNotebook(this.panel, runcode2, expr2, false);
                    }
                    let runcode3 = `with open("${this.tmp_log_path}", 'a' ) as f:\n       f.write("${text}")`;
                    let expr3 = { path: "" };
                    let res_tmp = _utils__WEBPACK_IMPORTED_MODULE_1__["default"].sendKernelRequestFromNotebook(this.panel, runcode3, expr3, false);
                    res_tmp.then(value => {
                        if (formatter === 'pytorch_inc_bf16') {
                            let read_log = `import re\nwith open("${this.tmp_log_path}", 'r') as f:\n    logs = f.readlines()\n    fps_list=[]\n    for log_line in logs[-4:]:\n        pat = re.compile(r\'\\d+\\.?\\d+')\n        fps = re.findall(pat,log_line)[-1]\n        fps_list.append(float(fps))\nmaxi = max(fps_list)\nindex = fps_list.index(maxi)\nboost = round(maxi/fps_list[0],1)\nfeatures=['','pytorch_inc_static_quant_fx','pytorch_inc_dynamic_quant','pytorch_inc_bf16']\nfeature_name=['Original Model','INC Enable INT8 (Static)','INC Enable INT8 (Dynamic)','INC Enable BF16']\nbest_feature = features[index]\nbest_name = feature_name[index]\nfeature_l = []\nfeature_l.append(best_feature)\nfrom neural_coder import enable\nenable(code="${this.tmp_path}",features=feature_l, overwrite=True)\nwith open("${this.tmp_path}", 'r') as f:\n    optimized_code = f.read()\n`;
                            let read_expr = { boost: "boost", best_feature: "best_feature", best_name: "best_name", optimizeCode: "optimized_code", feature_l: "fps_list", maxi: "maxi", index: "index" };
                            let read_result = _utils__WEBPACK_IMPORTED_MODULE_1__["default"].sendKernelRequestFromNotebook(this.panel, read_log, read_expr, false);
                            read_result.then(value => {
                                var _a, _b, _c, _d;
                                console.log("resres", value);
                                let boost = Object.values(value.boost.data)[0];
                                let best_name = Object.values(value.best_name.data)[0];
                                let optimizedTexts = Object.values(value.optimizeCode.data)[0];
                                let optimizeCodes = optimizedTexts.split('# this is the beginning of a single code snippet\\n').slice(1);
                                if (this.markdown) {
                                    this.markdown.model.value.text += `[NeuralCoder INFO] The Best Intel Optimization: ${best_name}  \n`;
                                    this.markdown.model.value.text += `[NeuralCoder INFO] You can get up to ${boost}X performance boost.  \n`;
                                }
                                // cell.outputArea.node.innerText +=`[NeuralCoder INFO] The Best Intel Optimization: ${best_name}\n`
                                // cell.outputArea.node.innerText += `[NeuralCoder INFO] You can get up to ${boost}X performance boost.\n`
                                optimizeCodes[optimizeCodes.length - 1] = optimizeCodes[optimizeCodes.length - 1].slice(0, -3);
                                for (let i = 0; i < optimizeCodes.length; ++i) {
                                    const cell = this.cells[i];
                                    const currentTexts = this.cells.map(cell => cell.model.value.text);
                                    const currentText = currentTexts[i];
                                    let optimizedtext = optimizeCodes[i];
                                    optimizedtext = optimizedtext.replace(/\\'\\\\n\\'/g, "^^^");
                                    optimizedtext = optimizedtext.replace(/\\\\n"/g, "+++");
                                    optimizedtext = optimizedtext.replace(/\\\\n'/g, "+++");
                                    optimizedtext = optimizedtext.replace(/"\\\\n/g, "@@@");
                                    optimizedtext = optimizedtext.replace(/'\\\\n/g, "@@@");
                                    optimizedtext = optimizedtext.replace(/\\n/g, '\n');
                                    optimizedtext = optimizedtext.replace(/\\'/g, "'");
                                    optimizedtext = optimizedtext.replace(/\^\^\^/g, "'\\n'");
                                    optimizedtext = optimizedtext.replace(/\+\+\+/g, "\\n\"");
                                    optimizedtext = optimizedtext.replace(/\@\@\@/g, "\"\\n");
                                    if (cell.model.value.text === currentText) {
                                        cell.model.value.text = optimizedtext;
                                    }
                                }
                                if (this.markdown) {
                                    this.markdown.model.value.text += `[NeuralCoder INFO] HardWare: 4th Gen Intel Xeon Scalable processor with AMX \n`;
                                    this.markdown.model.value.text += `[NeuralCoder INFO] The log was saved to neural_coder_workspace\\NeuralCoder${this.rand}.log  \n`;
                                }
                                // let command = "lscpu | grep 'Model name'"
                                // let get_hardware = `import subprocess\nsubp = subprocess.Popen("${command}",shell=True,stdout=subprocess.PIPE,stderr=subprocess.PIPE,encoding="utf-8")\nsubp.wait(2)\nhardware = subp.communicate()[0].replace("Model name:","").strip()`
                                // let expr_hardware = {hardware: "hardware"}
                                // let hard_res = NotebookUtilities.sendKernelRequestFromNotebook(this.panel, get_hardware, expr_hardware,false);
                                // hard_res.then(value =>{
                                //   let hard = Object.values(value.hardware.data)[0] as string;
                                //   if(this.markdown){
                                //     this.markdown.model.value.text += `[NeuralCoder INFO] HardWare: ${hard}  \n`
                                //     this.markdown.model.value.text += `[NeuralCoder INFO] The log was saved to neural_coder_workspace\\NeuralCoder${this.rand}.log  \n`
                                //   }
                                // cell.outputArea.node.innerText += `[NeuralCoder INFO] HardWare: ${hard}\n`
                                //  })
                                //  cell.outputArea.node.innerText += `[NeuralCoder INFO] The log was saved to lab_workspace\\NeuralCoder${this.rand}.log\n`
                                const run_svg = document.createElement("svg");
                                run_svg.innerHTML = _constants__WEBPACK_IMPORTED_MODULE_2__.Constants.ICON_RUN;
                                (_d = (_c = (_b = (_a = run === null || run === void 0 ? void 0 : run.node.firstChild) === null || _a === void 0 ? void 0 : _a.firstChild) === null || _b === void 0 ? void 0 : _b.firstChild) === null || _c === void 0 ? void 0 : _c.firstChild) === null || _d === void 0 ? void 0 : _d.replaceWith(run_svg);
                            });
                        }
                    });
                });
            }
        }
    }
}
class JupyterlabNotebookCodeOptimizer extends JupyterlabCodeOptimizer {
    constructor(notebookTracker, panel) {
        super(panel);
        this.notebookTracker = notebookTracker;
        this.notebookname = '';
    }
    async optimizeAction(config, formatter) {
        return this.optimizeCells(true, config, formatter);
    }
    async optimizeAllCodeCells(config, formatter, notebook, run) {
        return this.optimizeCells(false, config, formatter, notebook, run);
    }
    getCodeCells(ifmarkdown = true, notebook) {
        if (!this.notebookTracker.currentWidget) {
            return [];
        }
        const codeCells = [];
        notebook = notebook || this.notebookTracker.currentWidget.content;
        this.notebookname = notebook.title.label;
        let count = 0;
        notebook.widgets.forEach((cell) => {
            if (cell.model.type === 'code') {
                count += 1;
                codeCells.push(cell);
            }
        });
        if (ifmarkdown) {
            _jupyterlab_notebook__WEBPACK_IMPORTED_MODULE_0__.NotebookActions.insertBelow(notebook);
            this.notebookTracker.currentWidget.content.activeCellIndex = count + 1;
            _jupyterlab_notebook__WEBPACK_IMPORTED_MODULE_0__.NotebookActions.changeCellType(notebook, 'markdown');
            const activeCell = notebook.activeCell;
            if (activeCell) {
                this.markdown = activeCell;
            }
        }
        this.cells = codeCells;
        return codeCells;
    }
    async optimizeCells(selectedOnly, config, formatter, notebook, run) {
        if (this.working) {
            return new Promise((resolve, reject) => {
                resolve("false!");
            });
        }
        console.log("arrive here 333");
        this.working = true;
        const optimize_type = formatter !== undefined ? formatter : 'pytorch_mixed_precision_cpu';
        if (optimize_type === 'auto-quant') {
            selectedOnly = true;
        }
        else {
            selectedOnly = false;
        }
        const selectedCells = this.getCodeCells(selectedOnly, notebook);
        let cell = selectedCells[selectedCells.length - 1];
        if (selectedCells.length === 0) {
            this.working = false;
            return new Promise((resolve, reject) => {
                resolve("false!");
            });
        }
        const currentTexts = selectedCells.map(cell => cell.model.value.text);
        if (optimize_type === 'auto-quant') {
            console.log("arrive here 444-111");
            if (this.markdown) {
                this.markdown.model.value.text = `[NeuralCoder INFO] Auto-Quant Started ......  \n`;
                this.markdown.model.value.text += `[NeuralCoder INFO] Code: User code from Jupyter Lab notebook "${this.notebookname}"  \n`;
                this.markdown.model.value.text += `[NeuralCoder INFO] Benchmark Mode: Throughput  \n`;
            }
            // cell.outputArea.node.innerText = `[NeuralCoder INFO] Auto-Quant Started ......\n`
            // cell.outputArea.node.innerText += `[NeuralCoder INFO] Code: User code from Jupyter Lab notebook "${this.notebookname}"\n`
            // cell.outputArea.node.innerText += `[NeuralCoder INFO] Benchmark Mode: Throughput\n`
            let runcode = `with open('${this.log_path}', 'a' ) as f:\n       f.write("[NeuralCoder INFO] Auto-Quant Started ......\\n")`;
            let expr = { path: "" };
            _utils__WEBPACK_IMPORTED_MODULE_1__["default"].sendKernelRequestFromNotebook(this.panel, runcode, expr, false);
            let runcode2 = `with open('${this.log_path}', 'a' ) as f:\n       f.write("[NeuralCoder INFO] Code: User code from Jupyter Lab notebook '${this.notebookname}'\\n")`;
            let expr2 = { path: "" };
            _utils__WEBPACK_IMPORTED_MODULE_1__["default"].sendKernelRequestFromNotebook(this.panel, runcode2, expr2, false);
            let runcode3 = `with open('${this.log_path}', 'a' ) as f:\n       f.write("[NeuralCoder INFO] Benchmark Mode: Throughput\\n")`;
            let expr3 = { path: "" };
            _utils__WEBPACK_IMPORTED_MODULE_1__["default"].sendKernelRequestFromNotebook(this.panel, runcode3, expr3, false);
            // cell.outputArea.node.setAttribute("class","pad")
            await this.optimizeCode(currentTexts, '', 'The Original Model', 'INC Enable INT8 (Static)', config, true, this.panel, cell, run);
            await this.optimizeCode(currentTexts, 'pytorch_inc_static_quant_fx', 'INC Enable INT8 (Static)', 'INC Enable INT8 (Dynamic)', config, true, this.panel, cell, run);
            await this.optimizeCode(currentTexts, 'pytorch_inc_dynamic_quant', 'INC Enable INT8 (Dynamic)', 'INC Enable BF16', config, true, this.panel, cell, run);
            await this.optimizeCode(currentTexts, 'pytorch_inc_bf16', 'INC Enable BF16', '', config, true, this.panel, cell, run);
        }
        else {
            console.log("arrive here 444-222");
            await this.optimizeCode(currentTexts, optimize_type, "", "", "normal", true, this.panel, cell, run);
        }
        this.working = false;
        console.log("arrive here 555");
        return new Promise((resolve, reject) => {
            resolve("success!");
        });
    }
    applicable(formatter, currentWidget) {
        const currentNotebookWidget = this.notebookTracker.currentWidget;
        return currentNotebookWidget && currentWidget === currentNotebookWidget;
    }
}


/***/ }),

/***/ "./lib/index.js":
/*!**********************!*\
  !*** ./lib/index.js ***!
  \**********************/
/***/ ((__unused_webpack_module, __webpack_exports__, __webpack_require__) => {

__webpack_require__.r(__webpack_exports__);
/* harmony export */ __webpack_require__.d(__webpack_exports__, {
/* harmony export */   "default": () => (__WEBPACK_DEFAULT_EXPORT__)
/* harmony export */ });
/* harmony import */ var _jupyterlab_notebook__WEBPACK_IMPORTED_MODULE_0__ = __webpack_require__(/*! @jupyterlab/notebook */ "webpack/sharing/consume/default/@jupyterlab/notebook");
/* harmony import */ var _jupyterlab_notebook__WEBPACK_IMPORTED_MODULE_0___default = /*#__PURE__*/__webpack_require__.n(_jupyterlab_notebook__WEBPACK_IMPORTED_MODULE_0__);
/* harmony import */ var _jupyterlab_apputils__WEBPACK_IMPORTED_MODULE_1__ = __webpack_require__(/*! @jupyterlab/apputils */ "webpack/sharing/consume/default/@jupyterlab/apputils");
/* harmony import */ var _jupyterlab_apputils__WEBPACK_IMPORTED_MODULE_1___default = /*#__PURE__*/__webpack_require__.n(_jupyterlab_apputils__WEBPACK_IMPORTED_MODULE_1__);
/* harmony import */ var _jupyterlab_settingregistry__WEBPACK_IMPORTED_MODULE_2__ = __webpack_require__(/*! @jupyterlab/settingregistry */ "webpack/sharing/consume/default/@jupyterlab/settingregistry");
/* harmony import */ var _jupyterlab_settingregistry__WEBPACK_IMPORTED_MODULE_2___default = /*#__PURE__*/__webpack_require__.n(_jupyterlab_settingregistry__WEBPACK_IMPORTED_MODULE_2__);
/* harmony import */ var _jupyterlab_mainmenu__WEBPACK_IMPORTED_MODULE_3__ = __webpack_require__(/*! @jupyterlab/mainmenu */ "webpack/sharing/consume/default/@jupyterlab/mainmenu");
/* harmony import */ var _jupyterlab_mainmenu__WEBPACK_IMPORTED_MODULE_3___default = /*#__PURE__*/__webpack_require__.n(_jupyterlab_mainmenu__WEBPACK_IMPORTED_MODULE_3__);
/* harmony import */ var _jupyterlab_ui_components__WEBPACK_IMPORTED_MODULE_4__ = __webpack_require__(/*! @jupyterlab/ui-components */ "webpack/sharing/consume/default/@jupyterlab/ui-components");
/* harmony import */ var _jupyterlab_ui_components__WEBPACK_IMPORTED_MODULE_4___default = /*#__PURE__*/__webpack_require__.n(_jupyterlab_ui_components__WEBPACK_IMPORTED_MODULE_4__);
/* harmony import */ var _lumino_widgets__WEBPACK_IMPORTED_MODULE_5__ = __webpack_require__(/*! @lumino/widgets */ "webpack/sharing/consume/default/@lumino/widgets");
/* harmony import */ var _lumino_widgets__WEBPACK_IMPORTED_MODULE_5___default = /*#__PURE__*/__webpack_require__.n(_lumino_widgets__WEBPACK_IMPORTED_MODULE_5__);
/* harmony import */ var _deepcoder__WEBPACK_IMPORTED_MODULE_6__ = __webpack_require__(/*! ./deepcoder */ "./lib/deepcoder.js");
/* harmony import */ var _constants__WEBPACK_IMPORTED_MODULE_7__ = __webpack_require__(/*! ./constants */ "./lib/constants.js");








class neural_compressor_ext_lab {
    constructor(app, tracker, notebookpanel) {
        this.app = app;
        this.tracker = tracker;
        this.notebookpanel = notebookpanel;
        this.setupWidgetExtension();
        this.config = '';
    }
    createNew(nb) {
        this.notebookpanel = nb;
        this.notebookCodeOptimizer = new _deepcoder__WEBPACK_IMPORTED_MODULE_6__.JupyterlabNotebookCodeOptimizer(this.tracker, this.notebookpanel);
        const svg = document.createElement("svg");
        svg.innerHTML = _constants__WEBPACK_IMPORTED_MODULE_7__.Constants.ICON_FORMAT_ALL_SVG;
        const run_svg = document.createElement("svg");
        run_svg.innerHTML = _constants__WEBPACK_IMPORTED_MODULE_7__.Constants.ICON_RUN;
        const div = document.createElement("div");
        div.setAttribute("class", "wrapper");
        const span = document.createElement("span");
        span.setAttribute("class", "f1ozlkqi");
        span.innerHTML = _constants__WEBPACK_IMPORTED_MODULE_7__.Constants.SVG;
        const selector = document.createElement("select");
        selector.setAttribute("class", "aselector");
        selector.id = "NeuralCoder";
        const option1 = document.createElement("option");
        option1.value = "pytorch_inc_static_quant_fx";
        option1.innerText = "INC Enable INT8 (Static)";
        option1.selected = true;
        const option2 = document.createElement("option");
        option2.value = "pytorch_inc_dynamic_quant";
        option2.innerText = "INC Enable INT8 (Dynamic)";
        const option3 = document.createElement("option");
        option3.value = "pytorch_inc_bf16";
        option3.innerText = "INC Enable BF16";
        const option4 = document.createElement("option");
        option4.value = "auto-quant";
        option4.innerText = "INC Auto Enable & Benchmark";
        selector.options.add(option1);
        selector.options.add(option2);
        selector.options.add(option3);
        selector.options.add(option4);
        div.appendChild(selector);
        div.appendChild(span);
        const selector_widget = new _lumino_widgets__WEBPACK_IMPORTED_MODULE_5__.Widget();
        selector_widget.node.appendChild(div);
        selector_widget.addClass("aselector");
        let notebookCodeOptimizer = this.notebookCodeOptimizer;
        let config = this.config;
        const dia_input = document.createElement("input");
        const dia_widget = new _lumino_widgets__WEBPACK_IMPORTED_MODULE_5__.Widget();
        dia_widget.node.appendChild(dia_input);
        dia_widget.addClass("dialog");
        const run_button = new _jupyterlab_apputils__WEBPACK_IMPORTED_MODULE_1__.ToolbarButton({
            tooltip: 'NeuralCoder',
            icon: new _jupyterlab_ui_components__WEBPACK_IMPORTED_MODULE_4__.LabIcon({
                name: "run",
                svgstr: _constants__WEBPACK_IMPORTED_MODULE_7__.Constants.ICON_RUN
            }),
            onClick: async function () {
                var _a, _b, _c, _d;
                console.log("arrive here 111");
                (_d = (_c = (_b = (_a = run_button.node.firstChild) === null || _a === void 0 ? void 0 : _a.firstChild) === null || _b === void 0 ? void 0 : _b.firstChild) === null || _c === void 0 ? void 0 : _c.firstChild) === null || _d === void 0 ? void 0 : _d.replaceWith(svg);
                if (selector.options[selector.selectedIndex].value === 'auto-quant') {
                    await (0,_jupyterlab_apputils__WEBPACK_IMPORTED_MODULE_1__.showDialog)({
                        title: 'Please input execute parameters:',
                        body: dia_widget,
                        buttons: [_jupyterlab_apputils__WEBPACK_IMPORTED_MODULE_1__.Dialog.okButton({ label: 'Confirm' })]
                    }).then(result => {
                        if (result.button.accept) {
                            config = dia_input.value;
                        }
                    });
                }
                console.log("arrive here 222");
                await notebookCodeOptimizer.optimizeAllCodeCells(config, selector.options[selector.selectedIndex].value, undefined, run_button);
            }
        });
        nb.toolbar.insertItem(11, "nc", run_button);
        nb.toolbar.insertItem(12, "selector", selector_widget);
    }
    setupWidgetExtension() {
        this.app.docRegistry.addWidgetExtension('Notebook', this);
    }
}
/**
 * Initialization data for the neural_compressor_ext_lab extension.
 */
const plugin = {
    id: 'neural_compressor_ext_lab:plugin',
    autoStart: true,
    requires: [_jupyterlab_notebook__WEBPACK_IMPORTED_MODULE_0__.INotebookTracker, _jupyterlab_mainmenu__WEBPACK_IMPORTED_MODULE_3__.IMainMenu],
    optional: [_jupyterlab_settingregistry__WEBPACK_IMPORTED_MODULE_2__.ISettingRegistry],
    activate: (app, tracker, notebookpanel) => {
        new neural_compressor_ext_lab(app, tracker, notebookpanel);
        console.log('JupyterLab extension neural_compressor_ext_lab is activated!');
    }
};
/* harmony default export */ const __WEBPACK_DEFAULT_EXPORT__ = (plugin);


/***/ }),

/***/ "./lib/utils.js":
/*!**********************!*\
  !*** ./lib/utils.js ***!
  \**********************/
/***/ ((__unused_webpack_module, __webpack_exports__, __webpack_require__) => {

__webpack_require__.r(__webpack_exports__);
/* harmony export */ __webpack_require__.d(__webpack_exports__, {
/* harmony export */   "default": () => (/* binding */ NotebookUtilities)
/* harmony export */ });
/* harmony import */ var _jupyterlab_apputils__WEBPACK_IMPORTED_MODULE_0__ = __webpack_require__(/*! @jupyterlab/apputils */ "webpack/sharing/consume/default/@jupyterlab/apputils");
/* harmony import */ var _jupyterlab_apputils__WEBPACK_IMPORTED_MODULE_0___default = /*#__PURE__*/__webpack_require__.n(_jupyterlab_apputils__WEBPACK_IMPORTED_MODULE_0__);
/* harmony import */ var react_sanitized_html__WEBPACK_IMPORTED_MODULE_1__ = __webpack_require__(/*! react-sanitized-html */ "webpack/sharing/consume/default/react-sanitized-html/react-sanitized-html");
/* harmony import */ var react_sanitized_html__WEBPACK_IMPORTED_MODULE_1___default = /*#__PURE__*/__webpack_require__.n(react_sanitized_html__WEBPACK_IMPORTED_MODULE_1__);
/* harmony import */ var react__WEBPACK_IMPORTED_MODULE_2__ = __webpack_require__(/*! react */ "webpack/sharing/consume/default/react");
/* harmony import */ var react__WEBPACK_IMPORTED_MODULE_2___default = /*#__PURE__*/__webpack_require__.n(react__WEBPACK_IMPORTED_MODULE_2__);
/*
 * Copyright 2019-2020 The Kale Authors
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 *      http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */

// @ts-ignore


class NotebookUtilities {
    /**
      * generate random number
      * @Min
      * @Max
      */
    static GetRandomNum(Min, Max) {
        let Range;
        Range = Max - Min;
        var Rand = Math.random();
        return (Min + Math.round(Rand * Range));
    }
    /**
     * Builds an HTML container by sanitizing a list of strings and converting
     * them in valid HTML
     * @param msg A list of string with HTML formatting
     * @returns a HTMLDivElement composed of a list of spans with formatted text
     */
    static buildDialogBody(msg) {
        return (react__WEBPACK_IMPORTED_MODULE_2__.createElement("div", null, msg.map((s, i) => {
            return (react__WEBPACK_IMPORTED_MODULE_2__.createElement(react__WEBPACK_IMPORTED_MODULE_2__.Fragment, { key: `msg-${i}` },
                react__WEBPACK_IMPORTED_MODULE_2__.createElement((react_sanitized_html__WEBPACK_IMPORTED_MODULE_1___default()), { allowedAttributes: { a: ['href'] }, allowedTags: ['b', 'i', 'em', 'strong', 'a', 'pre'], html: s }),
                react__WEBPACK_IMPORTED_MODULE_2__.createElement("br", null)));
        })));
    }
    /**
     * Opens a pop-up dialog in JupyterLab to display a simple message.
     * @param title The title for the message popup
     * @param msg The message as an array of strings
     * @param buttonLabel The label to use for the button. Default is 'OK'
     * @param buttonClassName The classname to give to the 'ok' button
     * @returns Promise<void> - A promise once the message is closed.
     */
    static async showMessage(title, msg, buttonLabel = 'Dismiss', buttonClassName = '') {
        const buttons = [
            _jupyterlab_apputils__WEBPACK_IMPORTED_MODULE_0__.Dialog.okButton({ label: buttonLabel, className: buttonClassName }),
        ];
        const messageBody = this.buildDialogBody(msg);
        await (0,_jupyterlab_apputils__WEBPACK_IMPORTED_MODULE_0__.showDialog)({ title, buttons, body: messageBody });
    }
    /**
     * Opens a pop-up dialog in JupyterLab to display a yes/no dialog.
     * @param title The title for the message popup
     * @param msg The message
     * @param acceptLabel The label to use for the accept button. Default is 'YES'
     * @param rejectLabel The label to use for the reject button. Default is 'NO'
     * @param yesButtonClassName The classname to give to the accept button.
     * @param noButtonClassName The  classname to give to the cancel button.
     * @returns Promise<void> - A promise once the message is closed.
     */
    static async showYesNoDialog(title, msg, acceptLabel = 'YES', rejectLabel = 'NO', yesButtonClassName = '', noButtonClassName = '') {
        const buttons = [
            _jupyterlab_apputils__WEBPACK_IMPORTED_MODULE_0__.Dialog.okButton({ label: acceptLabel, className: yesButtonClassName }),
            _jupyterlab_apputils__WEBPACK_IMPORTED_MODULE_0__.Dialog.cancelButton({ label: rejectLabel, className: noButtonClassName }),
        ];
        const messageBody = this.buildDialogBody(msg);
        const result = await (0,_jupyterlab_apputils__WEBPACK_IMPORTED_MODULE_0__.showDialog)({ title, buttons, body: messageBody });
        return result.button.label === acceptLabel;
    }
    /**
     * Opens a pop-up dialog in JupyterLab with various information and button
     * triggering reloading the page.
     * @param title The title for the message popup
     * @param msg The message
     * @param buttonLabel The label to use for the button. Default is 'Refresh'
     * @param buttonClassName The  classname to give to the 'refresh' button.
     * @returns Promise<void> - A promise once the message is closed.
     */
    static async showRefreshDialog(title, msg, buttonLabel = 'Refresh', buttonClassName = '') {
        await this.showMessage(title, msg, buttonLabel, buttonClassName);
        location.reload();
    }
    /**
     * @description Creates a new JupyterLab notebook for use by the application
     * @param command The command registry
     * @returns Promise<NotebookPanel> - A promise containing the notebook panel object that was created (if successful).
     */
    static async createNewNotebook(command) {
        const notebook = await command.execute('notebook:create-new', {
            activate: true,
            path: '',
            preferredLanguage: '',
        });
        await notebook.session.ready;
        return notebook;
    }
    /**
     * Safely saves the Jupyter notebook document contents to disk
     * @param notebookPanel The notebook panel containing the notebook to save
     */
    static async saveNotebook(notebookPanel) {
        if (notebookPanel) {
            await notebookPanel.context.ready;
            notebookPanel.context.save();
            return true;
        }
        return false;
    }
    /**
     * Convert the notebook contents to JSON
     * @param notebookPanel The notebook panel containing the notebook to serialize
     */
    static notebookToJSON(notebookPanel) {
        if (notebookPanel.content.model) {
            return notebookPanel.content.model.toJSON();
        }
        return null;
    }
    /**
     * @description Gets the value of a key from specified notebook's metadata.
     * @param notebookPanel The notebook to get meta data from.
     * @param key The key of the value.
     * @returns any -The value of the metadata. Returns null if the key doesn't exist.
     */
    static getMetaData(notebookPanel, key) {
        if (!notebookPanel) {
            throw new Error('The notebook is null or undefined. No meta data available.');
        }
        if (notebookPanel.model && notebookPanel.model.metadata.has(key)) {
            return notebookPanel.model.metadata.get(key);
        }
        return null;
    }
    /**
     * @description Sets the key value pair in the notebook's metadata.
     * If the key doesn't exists it will add one.
     * @param notebookPanel The notebook to set meta data in.
     * @param key The key of the value to create.
     * @param value The value to set.
     * @param save Default is false. Whether the notebook should be saved after the meta data is set.
     * Note: This function will not wait for the save to complete, it only sends a save request.
     * @returns The old value for the key, or undefined if it did not exist.
     */
    static setMetaData(notebookPanel, key, value, save = false) {
        var _a;
        if (!notebookPanel) {
            throw new Error('The notebook is null or undefined. No meta data available.');
        }
        const oldVal = (_a = notebookPanel.model) === null || _a === void 0 ? void 0 : _a.metadata.set(key, value);
        if (save) {
            this.saveNotebook(notebookPanel);
        }
        return oldVal;
    }
    // /**
    //  * Get a new Kernel, not tied to a Notebook
    //  * Source code here: https://github.com/jupyterlab/jupyterlab/tree/473348d25bcb258ca2f0c127dd8fb5b193217135/packages/services
    //  */
    // public static async createNewKernel() {
    //   // Get info about the available kernels and start a new one.
    //   let options: Kernel.IOptions = await Kernel.getSpecs().then(kernelSpecs => {
    //     // console.log('Default spec:', kernelSpecs.default);
    //     // console.log('Available specs', Object.keys(kernelSpecs.kernelspecs));
    //     // use the default name
    //     return { name: kernelSpecs.default };
    //   });
    //   return await Kernel.startNew(options).then(_kernel => {
    //     return _kernel;
    //   });
    // }
    // // TODO: We can use this context manager to execute commands inside a new kernel
    // //  and be sure that it will be disposed of at the end.
    // //  Another approach could be to create a kale_rpc Kernel, as a singleton,
    // //  created at startup. The only (possible) drawback is that we can not name
    // //  a kernel instance with a custom id/name, so when refreshing JupyterLab we would
    // //  not recognize the kernel. A solution could be to have a kernel spec dedicated to kale rpc calls.
    // public static async executeWithNewKernel(action: Function, args: any[] = []) {
    //   // create brand new kernel
    //   const _k = await this.createNewKernel();
    //   // execute action inside kernel
    //   const res = await action(_k, ...args);
    //   // close kernel
    //   _k.shutdown();
    //   // return result
    //   return res;
    // }
    /**
     * @description This function runs code directly in the notebook's kernel and then evaluates the
     * result and returns it as a promise.
     * @param kernel The kernel to run the code in.
     * @param runCode The code to run in the kernel.
     * @param userExpressions The expressions used to capture the desired info from the executed code.
     * @param runSilent Default is false. If true, kernel will execute as quietly as possible.
     * store_history will be set to false, and no broadcast on IOPUB channel will be made.
     * @param storeHistory Default is false. If true, the code executed will be stored in the kernel's history
     * and the counter which is shown in the cells will be incremented to reflect code was run.
     * @param allowStdIn Default is false. If true, code running in kernel can prompt user for input using
     * an input_request message.
     * @param stopOnError Default is false. If True, does not abort the execution queue, if an exception is encountered.
     * This allows the queued execution of multiple execute_requests, even if they generate exceptions.
     * @returns Promise<any> - A promise containing the execution results of the code as an object with
     * keys based on the user_expressions.
     * @example
     * //The code
     * const code = "a=123\nb=456\nsum=a+b";
     * //The user expressions
     * const expr = {sum: "sum",prod: "a*b",args:"[a,b,sum]"};
     * //Async function call (returns a promise)
     * sendKernelRequest(notebookPanel, code, expr,false);
     * //Result when promise resolves:
     * {
     *  sum:{status:"ok",data:{"text/plain":"579"},metadata:{}},
     *  prod:{status:"ok",data:{"text/plain":"56088"},metadata:{}},
     *  args:{status:"ok",data:{"text/plain":"[123, 456, 579]"}}
     * }
     * @see For more information on JupyterLab messages:
     * https://jupyter-client.readthedocs.io/en/latest/messaging.html#execution-results
     */
    static async sendKernelRequest(kernel, runCode, userExpressions, runSilent = false, storeHistory = false, allowStdIn = false, stopOnError = false) {
        if (!kernel) {
            throw new Error('Kernel is null or undefined.');
        }
        // Wait for kernel to be ready before sending request
        // await kernel.status;
        const message = await kernel.requestExecute({
            allow_stdin: allowStdIn,
            code: runCode,
            silent: runSilent,
            stop_on_error: stopOnError,
            store_history: storeHistory,
            user_expressions: userExpressions,
        }).done;
        const content = message.content;
        if (content.status !== 'ok') {
            // If response is not 'ok', throw contents as error, log code
            const msg = `Code caused an error:\n${runCode}`;
            console.error(msg);
            if (content.traceback) {
                content.traceback.forEach((line) => console.log(line.replace(/[\u001b\u009b][[()#;?]*(?:[0-9]{1,4}(?:;[0-9]{0,4})*)?[0-9A-ORZcf-nqry=><]/g, '')));
            }
            throw content;
        }
        // Return user_expressions of the content
        return content.user_expressions;
    }
    /**
     * Same as method sendKernelRequest but passing
     * a NotebookPanel instead of a Kernel
     */
    static async sendKernelRequestFromNotebook(notebookPanel, runCode, userExpressions, runSilent = false, storeHistory = false, allowStdIn = false, stopOnError = false) {
        var _a, _b, _c, _d;
        if (!notebookPanel) {
            throw new Error('Notebook is null or undefined.');
        }
        // Wait for notebook panel to be ready
        await notebookPanel.activate;
        await ((_a = notebookPanel.sessionContext) === null || _a === void 0 ? void 0 : _a.ready);
        console.log('get kernel', (_b = notebookPanel.sessionContext.session) === null || _b === void 0 ? void 0 : _b.kernel);
        return this.sendKernelRequest((_d = (_c = notebookPanel.sessionContext) === null || _c === void 0 ? void 0 : _c.session) === null || _d === void 0 ? void 0 : _d.kernel, runCode, userExpressions, runSilent, storeHistory, allowStdIn, stopOnError);
    }
}


/***/ })

}]);
//# sourceMappingURL=lib_index_js.2c3b18119886a0a82200.js.map