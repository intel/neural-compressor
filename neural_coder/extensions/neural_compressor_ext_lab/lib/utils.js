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
import { Dialog, showDialog } from '@jupyterlab/apputils';
// @ts-ignore
import SanitizedHTML from 'react-sanitized-html';
import * as React from 'react';
export default class NotebookUtilities {
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
        return (React.createElement("div", null, msg.map((s, i) => {
            return (React.createElement(React.Fragment, { key: `msg-${i}` },
                React.createElement(SanitizedHTML, { allowedAttributes: { a: ['href'] }, allowedTags: ['b', 'i', 'em', 'strong', 'a', 'pre'], html: s }),
                React.createElement("br", null)));
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
            Dialog.okButton({ label: buttonLabel, className: buttonClassName }),
        ];
        const messageBody = this.buildDialogBody(msg);
        await showDialog({ title, buttons, body: messageBody });
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
            Dialog.okButton({ label: acceptLabel, className: yesButtonClassName }),
            Dialog.cancelButton({ label: rejectLabel, className: noButtonClassName }),
        ];
        const messageBody = this.buildDialogBody(msg);
        const result = await showDialog({ title, buttons, body: messageBody });
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
