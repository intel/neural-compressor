"use strict";
(self["webpackChunkneural-compressor-ext-lab"] = self["webpackChunkneural-compressor-ext-lab"] || []).push([["lib_index_js"],{

/***/ "./lib/client.js":
/*!***********************!*\
  !*** ./lib/client.js ***!
  \***********************/
/***/ ((__unused_webpack_module, __webpack_exports__, __webpack_require__) => {

__webpack_require__.r(__webpack_exports__);
/* harmony export */ __webpack_require__.d(__webpack_exports__, {
/* harmony export */   "default": () => (__WEBPACK_DEFAULT_EXPORT__)
/* harmony export */ });
/* harmony import */ var _jupyterlab_coreutils__WEBPACK_IMPORTED_MODULE_0__ = __webpack_require__(/*! @jupyterlab/coreutils */ "webpack/sharing/consume/default/@jupyterlab/coreutils");
/* harmony import */ var _jupyterlab_coreutils__WEBPACK_IMPORTED_MODULE_0___default = /*#__PURE__*/__webpack_require__.n(_jupyterlab_coreutils__WEBPACK_IMPORTED_MODULE_0__);
/* harmony import */ var _jupyterlab_services__WEBPACK_IMPORTED_MODULE_1__ = __webpack_require__(/*! @jupyterlab/services */ "webpack/sharing/consume/default/@jupyterlab/services");
/* harmony import */ var _jupyterlab_services__WEBPACK_IMPORTED_MODULE_1___default = /*#__PURE__*/__webpack_require__.n(_jupyterlab_services__WEBPACK_IMPORTED_MODULE_1__);
/* harmony import */ var _constants__WEBPACK_IMPORTED_MODULE_2__ = __webpack_require__(/*! ./constants */ "./lib/constants.js");



class JupyterlabDeepCoderClient {
    request(path, method, body) {
        const settings = _jupyterlab_services__WEBPACK_IMPORTED_MODULE_1__.ServerConnection.makeSettings();
        const fullUrl = _jupyterlab_coreutils__WEBPACK_IMPORTED_MODULE_0__.URLExt.join(settings.baseUrl, _constants__WEBPACK_IMPORTED_MODULE_2__.Constants.SHORT_PLUGIN_NAME, path);
        return _jupyterlab_services__WEBPACK_IMPORTED_MODULE_1__.ServerConnection.makeRequest(fullUrl, {
            body,
            method,
            headers: new Headers({
                'Plugin-Version': _constants__WEBPACK_IMPORTED_MODULE_2__.Constants.PLUGIN_VERSION
            })
        }, settings).then(response => {
            if (response.status !== 200) {
                console.log("response:::", response.status);
                return response.text().then(() => {
                    throw new _jupyterlab_services__WEBPACK_IMPORTED_MODULE_1__.ServerConnection.ResponseError(response, response.statusText);
                });
            }
            return response.text();
        });
    }
}
/* harmony default export */ const __WEBPACK_DEFAULT_EXPORT__ = (JupyterlabDeepCoderClient);


/***/ }),

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
    Constants.SHORT_PLUGIN_NAME = 'neural-compressor-ext-lab';
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
/* harmony export */   "JupyterlabFileEditorCodeOptimizer": () => (/* binding */ JupyterlabFileEditorCodeOptimizer),
/* harmony export */   "JupyterlabNotebookCodeOptimizer": () => (/* binding */ JupyterlabNotebookCodeOptimizer)
/* harmony export */ });
/* harmony import */ var _jupyterlab_apputils__WEBPACK_IMPORTED_MODULE_0__ = __webpack_require__(/*! @jupyterlab/apputils */ "webpack/sharing/consume/default/@jupyterlab/apputils");
/* harmony import */ var _jupyterlab_apputils__WEBPACK_IMPORTED_MODULE_0___default = /*#__PURE__*/__webpack_require__.n(_jupyterlab_apputils__WEBPACK_IMPORTED_MODULE_0__);

class JupyterlabCodeOptimizer {
    constructor(client) {
        this.client = client;
        this.working = false;
    }
    optimizeCode(code, formatter, options, notebook) {
        return this.client
            .request('optimize', 'POST', JSON.stringify({
            code,
            notebook,
            formatter,
            options
        }))
            .then(resp => JSON.parse(resp));
    }
}
class JupyterlabNotebookCodeOptimizer extends JupyterlabCodeOptimizer {
    constructor(client, notebookTracker) {
        super(client);
        this.notebookTracker = notebookTracker;
    }
    async optimizeAction(config, formatter) {
        return this.optimizeCells(true, config, formatter);
    }
    async optimizeAllCodeCells(config, formatter, notebook) {
        return this.optimizeCells(false, config, formatter, notebook);
    }
    getCodeCells(selectedOnly = true, notebook) {
        if (!this.notebookTracker.currentWidget) {
            return [];
        }
        const codeCells = [];
        notebook = notebook || this.notebookTracker.currentWidget.content;
        notebook.widgets.forEach((cell) => {
            if (cell.model.type === 'code') {
                codeCells.push(cell);
            }
        });
        return codeCells;
    }
    async optimizeCells(selectedOnly, config, formatter, notebook) {
        var _a;
        if (this.working) {
            return;
        }
        try {
            this.working = true;
            const selectedCells = this.getCodeCells(selectedOnly, notebook);
            if (selectedCells.length === 0) {
                this.working = false;
                return;
            }
            const optimize_type = formatter !== undefined ? formatter : 'pytorch_mixed_precision_cpu';
            const currentTexts = selectedCells.map(cell => cell.model.value.text);
            console.log("arrive here 1");
            const optimizedTexts = await this.optimizeCode(currentTexts, optimize_type, undefined, true);
            console.log("arrive here 2");
            for (let i = 0; i < selectedCells.length; ++i) {
                const cell = selectedCells[i];
                const currentText = currentTexts[i];
                const optimizedText = optimizedTexts.code[i];
                if (cell.model.value.text === currentText) {
                    if (optimizedText.error) {
                        if (!((_a = config.suppressFormatterErrors) !== null && _a !== void 0 ? _a : false)) {
                            console.log("here 1");
                            await (0,_jupyterlab_apputils__WEBPACK_IMPORTED_MODULE_0__.showErrorMessage)('Optimize Code Error', optimizedText.error);
                        }
                    }
                    else {
                        cell.model.value.text = optimizedText;
                        cell.outputArea.node.innerText = "tothelighthouse";
                    }
                }
                else {
                    console.log("here 2");
                    await (0,_jupyterlab_apputils__WEBPACK_IMPORTED_MODULE_0__.showErrorMessage)('Optimize Code Error', `Cell value changed since format request was sent, formatting for cell ${i} skipped.`);
                }
            }
        }
        catch (error) {
            console.log("here 3");
            await (0,_jupyterlab_apputils__WEBPACK_IMPORTED_MODULE_0__.showErrorMessage)('Optimize Code Error', error);
        }
        this.working = false;
    }
    applicable(formatter, currentWidget) {
        const currentNotebookWidget = this.notebookTracker.currentWidget;
        return currentNotebookWidget && currentWidget === currentNotebookWidget;
    }
}
class JupyterlabFileEditorCodeOptimizer extends JupyterlabCodeOptimizer {
    constructor(client, editorTracker) {
        super(client);
        this.editorTracker = editorTracker;
    }
    optimizeAction(config, formatter) {
        if (this.working) {
            return;
        }
        const editorWidget = this.editorTracker.currentWidget;
        this.working = true;
        if (editorWidget == null) {
            return;
        }
        const editor = editorWidget.content.editor;
        const code = editor.model.value.text;
        this.optimizeCode([code], formatter, config[formatter], false)
            .then(data => {
            if (data.code[0].error) {
                console.log("here 4");
                void (0,_jupyterlab_apputils__WEBPACK_IMPORTED_MODULE_0__.showErrorMessage)('Optimize Code Error', data.code[0].error);
                this.working = false;
                return;
            }
            this.working = false;
        })
            .catch(error => {
            this.working = false;
            console.log("here 5");
            void (0,_jupyterlab_apputils__WEBPACK_IMPORTED_MODULE_0__.showErrorMessage)('Optimize Code Error', error);
        });
    }
    applicable(formatter, currentWidget) {
        const currentEditorWidget = this.editorTracker.currentWidget;
        return currentEditorWidget && currentWidget === currentEditorWidget;
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
/* harmony import */ var _deepcoder__WEBPACK_IMPORTED_MODULE_7__ = __webpack_require__(/*! ./deepcoder */ "./lib/deepcoder.js");
/* harmony import */ var _client__WEBPACK_IMPORTED_MODULE_6__ = __webpack_require__(/*! ./client */ "./lib/client.js");
/* harmony import */ var _constants__WEBPACK_IMPORTED_MODULE_8__ = __webpack_require__(/*! ./constants */ "./lib/constants.js");




// import { DisposableDelegate, IDisposable } from '@lumino/disposable';





class JupyterLabDeepCoder {
    // private panel: NotebookPanel;
    constructor(app, tracker) {
        this.app = app;
        this.tracker = tracker;
        // this.panel = panel;
        this.client = new _client__WEBPACK_IMPORTED_MODULE_6__["default"]();
        this.notebookCodeOptimizer = new _deepcoder__WEBPACK_IMPORTED_MODULE_7__.JupyterlabNotebookCodeOptimizer(this.client, this.tracker);
        this.setupWidgetExtension();
    }
    createNew(nb) {
        // this.panel = nb;
        const svg = document.createElement("svg");
        svg.innerHTML = _constants__WEBPACK_IMPORTED_MODULE_8__.Constants.ICON_FORMAT_ALL_SVG;
        const run_svg = document.createElement("svg");
        run_svg.innerHTML = _constants__WEBPACK_IMPORTED_MODULE_8__.Constants.ICON_RUN;
        const div = document.createElement("div");
        div.setAttribute("class", "wrapper");
        const span = document.createElement("span");
        span.setAttribute("class", "f1ozlkqi");
        span.innerHTML = _constants__WEBPACK_IMPORTED_MODULE_8__.Constants.SVG;
        const selector = document.createElement("select");
        selector.setAttribute("class", "aselector");
        selector.id = "NeuralCoder";
        const option1 = document.createElement("option");
        option1.value = "pytorch_inc_static_quant_fx";
        option1.innerText = "Intel INT8 (Static)";
        option1.selected = true;
        const option2 = document.createElement("option");
        option2.value = "pytorch_inc_dynamic_quant";
        option2.innerText = "Intel INT8 (Dynamic)";
        const option3 = document.createElement("option");
        option3.value = "pytorch_inc_bf16";
        option3.innerText = "Intel BF16";
        const option4 = document.createElement("option");
        option4.value = "auto-quant";
        option4.innerText = "Auto";
        selector.options.add(option1);
        selector.options.add(option2);
        selector.options.add(option3);
        selector.options.add(option4);
        div.appendChild(selector);
        div.appendChild(span);
        const selector_widget = new _lumino_widgets__WEBPACK_IMPORTED_MODULE_5__.Widget();
        selector_widget.node.appendChild(div);
        selector_widget.addClass("aselector");
        // let panel = this.panel;
        let notebookCodeOptimizer = this.notebookCodeOptimizer;
        let config = this.config;
        const run_button = new _jupyterlab_apputils__WEBPACK_IMPORTED_MODULE_1__.ToolbarButton({
            tooltip: 'NeuralCoder',
            icon: new _jupyterlab_ui_components__WEBPACK_IMPORTED_MODULE_4__.LabIcon({
                name: "run",
                svgstr: _constants__WEBPACK_IMPORTED_MODULE_8__.Constants.ICON_RUN
            }),
            onClick: async function () {
                var _a, _b, _c, _d, _e, _f, _g, _h;
                (_d = (_c = (_b = (_a = run_button.node.firstChild) === null || _a === void 0 ? void 0 : _a.firstChild) === null || _b === void 0 ? void 0 : _b.firstChild) === null || _c === void 0 ? void 0 : _c.firstChild) === null || _d === void 0 ? void 0 : _d.replaceWith(svg);
                console.log("user's selecting feature");
                console.log(selector.options[selector.selectedIndex].value);
                await notebookCodeOptimizer.optimizeAllCodeCells(config, selector.options[selector.selectedIndex].value);
                (_h = (_g = (_f = (_e = run_button.node.firstChild) === null || _e === void 0 ? void 0 : _e.firstChild) === null || _f === void 0 ? void 0 : _f.firstChild) === null || _g === void 0 ? void 0 : _g.firstChild) === null || _h === void 0 ? void 0 : _h.replaceWith(run_svg);
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
 * Initialization data for the deepcoder-jupyterlab extension.
 */
const plugin = {
    id: 'deepcoder-jupyterlab:plugin',
    autoStart: true,
    requires: [_jupyterlab_notebook__WEBPACK_IMPORTED_MODULE_0__.INotebookTracker, _jupyterlab_mainmenu__WEBPACK_IMPORTED_MODULE_3__.IMainMenu],
    optional: [_jupyterlab_settingregistry__WEBPACK_IMPORTED_MODULE_2__.ISettingRegistry],
    activate: (app, tracker) => {
        new JupyterLabDeepCoder(app, tracker);
        console.log('JupyterLab extension jupyterlab_neuralcoder is activated!');
    }
};
/* harmony default export */ const __WEBPACK_DEFAULT_EXPORT__ = (plugin);


/***/ })

}]);
//# sourceMappingURL=lib_index_js.c8ce5d6f06928ba82660.js.map