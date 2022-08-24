import { INotebookTracker, } from '@jupyterlab/notebook';
import { 
// ICommandPalette,
ToolbarButton, } from '@jupyterlab/apputils';
import { ISettingRegistry } from '@jupyterlab/settingregistry';
import { IMainMenu } from '@jupyterlab/mainmenu';
// import { DisposableDelegate, IDisposable } from '@lumino/disposable';
import { LabIcon } from '@jupyterlab/ui-components';
import { Widget } from '@lumino/widgets';
import { JupyterlabNotebookCodeOptimizer } from './deepcoder';
import JupyterlabDeepCoderClient from './client';
import { Constants } from './constants';
class JupyterLabDeepCoder {
    // private panel: NotebookPanel;
    constructor(app, tracker) {
        this.app = app;
        this.tracker = tracker;
        // this.panel = panel;
        this.client = new JupyterlabDeepCoderClient();
        this.notebookCodeOptimizer = new JupyterlabNotebookCodeOptimizer(this.client, this.tracker);
        this.setupWidgetExtension();
    }
    createNew(nb) {
        // this.panel = nb;
        const svg = document.createElement("svg");
        svg.innerHTML = Constants.ICON_FORMAT_ALL_SVG;
        const run_svg = document.createElement("svg");
        run_svg.innerHTML = Constants.ICON_RUN;
        const div = document.createElement("div");
        div.setAttribute("class", "wrapper");
        const span = document.createElement("span");
        span.setAttribute("class", "f1ozlkqi");
        span.innerHTML = Constants.SVG;
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
        selector.options.add(option1);
        selector.options.add(option2);
        selector.options.add(option3);
        selector.options.add(option4);
        div.appendChild(selector);
        div.appendChild(span);
        const selector_widget = new Widget();
        selector_widget.node.appendChild(div);
        selector_widget.addClass("aselector");
        // let panel = this.panel;
        let notebookCodeOptimizer = this.notebookCodeOptimizer;
        let config = this.config;
        const run_button = new ToolbarButton({
            tooltip: 'NeuralCoder',
            icon: new LabIcon({
                name: "run",
                svgstr: Constants.ICON_RUN
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
    id: 'neural-compressor-ext-lab:plugin',
    autoStart: true,
    requires: [INotebookTracker, IMainMenu],
    optional: [ISettingRegistry],
    activate: (app, tracker) => {
        new JupyterLabDeepCoder(app, tracker);
        console.log('JupyterLab extension jupyterlab_neuralcoder is activated!');
    }
};
export default plugin;
