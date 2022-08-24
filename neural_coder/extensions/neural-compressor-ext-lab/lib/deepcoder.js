import { showErrorMessage } from '@jupyterlab/apputils';
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
export class JupyterlabNotebookCodeOptimizer extends JupyterlabCodeOptimizer {
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
                            await showErrorMessage('Optimize Code Error', optimizedText.error);
                        }
                    }
                    else {
                        cell.model.value.text = optimizedText;
                        cell.outputArea.node.innerText = "tothelighthouse";
                    }
                }
                else {
                    await showErrorMessage('Optimize Code Error', `Cell value changed since format request was sent, formatting for cell ${i} skipped.`);
                }
            }
        }
        catch (error) {
            await showErrorMessage('Optimize Code Error', error);
        }
        this.working = false;
    }
    applicable(formatter, currentWidget) {
        const currentNotebookWidget = this.notebookTracker.currentWidget;
        return currentNotebookWidget && currentWidget === currentNotebookWidget;
    }
}
export class JupyterlabFileEditorCodeOptimizer extends JupyterlabCodeOptimizer {
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
                void showErrorMessage('Optimize Code Error', data.code[0].error);
                this.working = false;
                return;
            }
            this.working = false;
        })
            .catch(error => {
            this.working = false;
            void showErrorMessage('Optimize Code Error', error);
        });
    }
    applicable(formatter, currentWidget) {
        const currentEditorWidget = this.editorTracker.currentWidget;
        return currentEditorWidget && currentWidget === currentEditorWidget;
    }
}
