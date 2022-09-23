import { DocumentRegistry } from '@jupyterlab/docregistry';
import {
  JupyterFrontEnd,
  JupyterFrontEndPlugin
} from '@jupyterlab/application';
import { INotebookTracker,NotebookPanel, INotebookModel} from '@jupyterlab/notebook';
import { 
ToolbarButton, showDialog, Dialog } from '@jupyterlab/apputils';
import { ISettingRegistry } from '@jupyterlab/settingregistry';
import { IMainMenu } from '@jupyterlab/mainmenu';
import { LabIcon } from '@jupyterlab/ui-components';
import { Widget } from '@lumino/widgets';
import { JupyterlabNotebookCodeOptimizer } from './deepcoder';
import { Constants } from './constants';


class neural_compressor_ext_lab
  implements DocumentRegistry.IWidgetExtension<NotebookPanel, INotebookModel> {
  private app: JupyterFrontEnd;
  private tracker: INotebookTracker;
  private notebookpanel: NotebookPanel;
  private config: string;
  private notebookCodeOptimizer: JupyterlabNotebookCodeOptimizer | undefined;

  constructor(
    app: JupyterFrontEnd,
    tracker: INotebookTracker,
    notebookpanel:NotebookPanel
  ) {
    this.app = app;
    this.tracker = tracker;
    this.notebookpanel = notebookpanel;
    this.setupWidgetExtension();
    this.config = ''
  }

  public createNew(
    nb: NotebookPanel,
  ) {
    this.notebookpanel = nb;
    this.notebookCodeOptimizer = new JupyterlabNotebookCodeOptimizer(
    this.tracker,
    this.notebookpanel
  ); 
    const svg = document.createElement("svg")
    svg.innerHTML = Constants.ICON_FORMAT_ALL_SVG
    const run_svg = document.createElement("svg")
    run_svg.innerHTML = Constants.ICON_RUN
    const div = document.createElement("div");
    div.setAttribute("class","wrapper")
    const span = document.createElement("span");
    span.setAttribute("class","f1ozlkqi")
    span.innerHTML = Constants.SVG;
    const selector = document.createElement("select");
    selector.setAttribute("class","aselector")
    selector.id = "NeuralCoder"
    const option1 = document.createElement("option");
    option1.value = "pytorch_aliblade";
    option1.innerText = "Alibaba Blade-DISC";
    const option2 = document.createElement("option");
    option2.value = "auto";
    option2.innerText = "Auto Benchmark";
    option1.selected=true;
    
    selector.options.add(option1)
    selector.options.add(option2)
    div.appendChild(selector)
    div.appendChild(span)
    const selector_widget = new Widget();
    selector_widget.node.appendChild(div)
    selector_widget.addClass("aselector")
    let notebookCodeOptimizer = this.notebookCodeOptimizer;
    let config = this.config;
    
    const dia_input = document.createElement("input")
    const dia_widget = new Widget();
    dia_widget.node.appendChild(dia_input)
    dia_widget.addClass("dialog")

    const run_button = new ToolbarButton({
      tooltip: 'NeuralCoder',
      icon: new LabIcon({
        name: "run",
        svgstr:Constants.ICON_RUN
      }),
      onClick: async function (){  
        run_button.node.firstChild?.firstChild?.firstChild?.firstChild?.replaceWith(svg)
        if (selector.options[selector.selectedIndex].value === 'auto'){
            await showDialog({
              title:'Please input execute parameters:',
              body: dia_widget,
              buttons: [Dialog.okButton({ label: 'Confirm' })]
            }).then(result => {
               if (result.button.accept) {
                 config = dia_input.value
               }
            })
          }
        await notebookCodeOptimizer.optimizeAllCodeCells(config,selector.options[selector.selectedIndex].value,undefined,run_button);
      }
    });
    nb.toolbar.insertItem(11,"nc",run_button)
    nb.toolbar.insertItem(12,"selector",selector_widget)

  }
 
  private setupWidgetExtension() {
    this.app.docRegistry.addWidgetExtension('Notebook', this);
  }
}



/**
 * Initialization data for the neural_compressor_ext_lab extension.
 */
const plugin: JupyterFrontEndPlugin<void> = {
  id: 'neural_compressor_ext_lab:plugin',
  autoStart: true,
  requires: [INotebookTracker,IMainMenu],
  optional: [ISettingRegistry],
  activate: (
    app: JupyterFrontEnd,
    tracker: INotebookTracker,
    notebookpanel: NotebookPanel
  ) => {
    new neural_compressor_ext_lab(
      app,
      tracker,
      notebookpanel
    );
    console.log('JupyterLab extension neural_compressor_ext_lab is activated!');
  }
};

export default plugin;
