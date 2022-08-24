import { INotebookTracker, Notebook } from '@jupyterlab/notebook';
import { IEditorTracker } from '@jupyterlab/fileeditor';
import { Widget } from '@lumino/widgets';
import JupyterlabDeepCoderClient from './client';
declare class JupyterlabCodeOptimizer {
    protected client: JupyterlabDeepCoderClient;
    protected working: boolean;
    constructor(client: JupyterlabDeepCoderClient);
    protected optimizeCode(code: string[], formatter: string, options: any, notebook: boolean): Promise<any>;
}
export declare class JupyterlabNotebookCodeOptimizer extends JupyterlabCodeOptimizer {
    protected notebookTracker: INotebookTracker;
    constructor(client: JupyterlabDeepCoderClient, notebookTracker: INotebookTracker);
    optimizeAction(config: any, formatter?: string): Promise<void>;
    optimizeAllCodeCells(config: any, formatter?: string, notebook?: Notebook): Promise<void>;
    private getCodeCells;
    private optimizeCells;
    applicable(formatter: string, currentWidget: Widget): boolean | null;
}
export declare class JupyterlabFileEditorCodeOptimizer extends JupyterlabCodeOptimizer {
    protected editorTracker: IEditorTracker;
    constructor(client: JupyterlabDeepCoderClient, editorTracker: IEditorTracker);
    optimizeAction(config: any, formatter: string): void;
    applicable(formatter: string, currentWidget: Widget): boolean | null;
}
export {};
