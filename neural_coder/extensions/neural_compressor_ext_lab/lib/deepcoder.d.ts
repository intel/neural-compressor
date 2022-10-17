import { Cell, CodeCell } from '@jupyterlab/cells';
import { ToolbarButton } from '@jupyterlab/apputils';
import { Widget } from '@lumino/widgets';
import { INotebookTracker, NotebookPanel, Notebook } from '@jupyterlab/notebook';
declare class JupyterlabCodeOptimizer {
    protected working: boolean;
    protected panel: NotebookPanel;
    private tmp_path;
    log_path: string;
    tmp_log_path: string;
    rand: number;
    markdown: Cell | undefined;
    cells: CodeCell[];
    constructor(panel: NotebookPanel);
    optimizeCode(code: string[], formatter: string, name: string, next: string, options: string | undefined, notebook: boolean, panel: NotebookPanel, cell: CodeCell, run?: ToolbarButton | undefined): Promise<void>;
}
export declare class JupyterlabNotebookCodeOptimizer extends JupyterlabCodeOptimizer {
    protected notebookname: string;
    protected notebookTracker: INotebookTracker;
    constructor(notebookTracker: INotebookTracker, panel: NotebookPanel);
    optimizeAction(config: any, formatter?: string): Promise<string>;
    optimizeAllCodeCells(config?: string, formatter?: string, notebook?: Notebook, run?: ToolbarButton): Promise<string>;
    private getCodeCells;
    private optimizeCells;
    applicable(formatter: string, currentWidget: Widget): boolean | null;
}
export {};
