import { INotebookTracker } from '@jupyterlab/notebook';
import { ToolbarButton, showDialog, Dialog } from '@jupyterlab/apputils';
import { ISettingRegistry } from '@jupyterlab/settingregistry';
import { IMainMenu } from '@jupyterlab/mainmenu';
import { LabIcon } from '@jupyterlab/ui-components';
import { Widget } from '@lumino/widgets';
import { JupyterlabNotebookCodeOptimizer } from './deepcoder';
import { Constants } from './constants';
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
        this.notebookCodeOptimizer = new JupyterlabNotebookCodeOptimizer(this.tracker, this.notebookpanel);
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
        const selector_widget = new Widget();
        selector_widget.node.appendChild(div);
        selector_widget.addClass("aselector");
        let notebookCodeOptimizer = this.notebookCodeOptimizer;
        let config = this.config;
        const dia_input = document.createElement("input");
        const dia_widget = new Widget();
        dia_widget.node.appendChild(dia_input);
        dia_widget.addClass("dialog");
        const run_button = new ToolbarButton({
            tooltip: 'NeuralCoder',
            icon: new LabIcon({
                name: "run",
                svgstr: Constants.ICON_RUN
            }),
            onClick: async function () {
                var _a, _b, _c, _d;
                console.log("arrive here 111");
                (_d = (_c = (_b = (_a = run_button.node.firstChild) === null || _a === void 0 ? void 0 : _a.firstChild) === null || _b === void 0 ? void 0 : _b.firstChild) === null || _c === void 0 ? void 0 : _c.firstChild) === null || _d === void 0 ? void 0 : _d.replaceWith(svg);
                if (selector.options[selector.selectedIndex].value === 'auto-quant') {
                    await showDialog({
                        title: 'Please input execute parameters:',
                        body: dia_widget,
                        buttons: [Dialog.okButton({ label: 'Confirm' })]
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
    requires: [INotebookTracker, IMainMenu],
    optional: [ISettingRegistry],
    activate: (app, tracker, notebookpanel) => {
        new neural_compressor_ext_lab(app, tracker, notebookpanel);
        console.log('JupyterLab extension neural_compressor_ext_lab is activated!');
    }
};
export default plugin;
