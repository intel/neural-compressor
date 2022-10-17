import { NotebookActions } from '@jupyterlab/notebook';
import NotebookUtilities from "./utils";
import { Constants } from './constants';
class JupyterlabCodeOptimizer {
    constructor(panel) {
        this.working = false;
        this.panel = panel;
        this.tmp_path = "tmp.py";
        this.rand = NotebookUtilities.GetRandomNum(0, 200);
        this.log_path = Constants.WORK_PATH + "NeuralCoder" + this.rand + ".log";
        this.tmp_log_path = Constants.WORK_PATH + "NeuralCoder_tmp" + ".log";
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
        NotebookUtilities.sendKernelRequestFromNotebook(panel, gen_code, expr, false);
        if (options === 'normal') {
            let runcode = `from neural_coder import enable\nenable(code="${this.tmp_path}",features=["${formatter}"], overwrite=True)`;
            let expr = { sum: ` ` };
            NotebookUtilities.sendKernelRequestFromNotebook(panel, runcode, expr, false);
            let run_code1 = `with open("${this.tmp_path}", 'r') as f:\n    optimized_code = f.read()\n`;
            let expr1 = { optimizedCode: "optimized_code" };
            let result2 = NotebookUtilities.sendKernelRequestFromNotebook(panel, run_code1, expr1, false);
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
                    run_svg.innerHTML = Constants.ICON_RUN;
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
                NotebookUtilities.sendKernelRequestFromNotebook(panel, runcode1, expr1, false);
                let runcode = `from neural_coder import enable\nperfomance, mode, path = enable(code="${this.tmp_path}",features=[], run_bench=True, args="${options}")\nwith open(path + '/bench.log', 'r') as f:\n    logs = f.readlines()\nlog_line = logs[4]\nlog = log_line.split("[")[1].split("]")[0]`;
                let expr = { path: "path", log: "log" };
                let result = NotebookUtilities.sendKernelRequestFromNotebook(panel, runcode, expr, false);
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
                    NotebookUtilities.sendKernelRequestFromNotebook(this.panel, runcode, expr, false);
                    if (this.markdown) {
                        this.markdown.model.value.text += `[NeuralCoder INFO] Enabling and Benchmarking for ${next} ......  \n`;
                    }
                    // cell.outputArea.node.innerText += `[NeuralCoder INFO] Enabling and Benchmarking for ${next} ......\n`
                    let runcode1 = `with open("${this.log_path}", 'a' ) as f:\n       f.write("[NeuralCoder INFO] Enabling and Benchmarking for ${next} ......\\n")`;
                    let expr1 = { path: "" };
                    NotebookUtilities.sendKernelRequestFromNotebook(panel, runcode1, expr1, false);
                    let runcode2 = `with open("${this.tmp_log_path}", 'a' ) as f:\n       f.write("${text}")`;
                    let expr2 = { path: "" };
                    NotebookUtilities.sendKernelRequestFromNotebook(this.panel, runcode2, expr2, false);
                });
            }
            else {
                let runcode = `from neural_coder import enable\nperfomance, mode, path = enable(code="${this.tmp_path}", features=["${formatter}"], run_bench=True, args="${options}")\nwith open(path + '/bench.log', 'r') as f:\n    logs = f.readlines()\nlog_line = logs[4]\nlog = log_line.split("[")[1].split("]")[0]`;
                let expr = { path: "path", log: "log" };
                let result = NotebookUtilities.sendKernelRequestFromNotebook(panel, runcode, expr, false);
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
                    NotebookUtilities.sendKernelRequestFromNotebook(this.panel, runcode, expr, false);
                    if (next !== '') {
                        if (this.markdown) {
                            this.markdown.model.value.text += `[NeuralCoder INFO] Enabling and Benchmarking for ${next} ......  \n`;
                        }
                        // cell.outputArea.node.innerText += `[NeuralCoder INFO] Enabling and Benchmarking for ${next} ......\n`
                        let runcode2 = `with open("${this.log_path}", 'a' ) as f:\n       f.write("[NeuralCoder INFO] Enabling and Benchmarking for ${next} ......\\n")`;
                        let expr2 = { path: "" };
                        NotebookUtilities.sendKernelRequestFromNotebook(this.panel, runcode2, expr2, false);
                    }
                    let runcode3 = `with open("${this.tmp_log_path}", 'a' ) as f:\n       f.write("${text}")`;
                    let expr3 = { path: "" };
                    let res_tmp = NotebookUtilities.sendKernelRequestFromNotebook(this.panel, runcode3, expr3, false);
                    res_tmp.then(value => {
                        if (formatter === 'pytorch_inc_bf16') {
                            let read_log = `import re\nwith open("${this.tmp_log_path}", 'r') as f:\n    logs = f.readlines()\n    fps_list=[]\n    for log_line in logs[-4:]:\n        pat = re.compile(r\'\\d+\\.?\\d+')\n        fps = re.findall(pat,log_line)[-1]\n        fps_list.append(float(fps))\nmaxi = max(fps_list)\nindex = fps_list.index(maxi)\nboost = round(maxi/fps_list[0],1)\nfeatures=['','pytorch_inc_static_quant_fx','pytorch_inc_dynamic_quant','pytorch_inc_bf16']\nfeature_name=['Original Model','INC Enable INT8 (Static)','INC Enable INT8 (Dynamic)','INC Enable BF16']\nbest_feature = features[index]\nbest_name = feature_name[index]\nfeature_l = []\nfeature_l.append(best_feature)\nfrom neural_coder import enable\nenable(code="${this.tmp_path}",features=feature_l, overwrite=True)\nwith open("${this.tmp_path}", 'r') as f:\n    optimized_code = f.read()\n`;
                            let read_expr = { boost: "boost", best_feature: "best_feature", best_name: "best_name", optimizeCode: "optimized_code", feature_l: "fps_list", maxi: "maxi", index: "index" };
                            let read_result = NotebookUtilities.sendKernelRequestFromNotebook(this.panel, read_log, read_expr, false);
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
                                // if(this.markdown){
                                //       this.markdown.model.value.text += `[NeuralCoder INFO] HardWare: 4th Gen Intel Xeon Scalable processor with AMX \n`
                                //       this.markdown.model.value.text += `[NeuralCoder INFO] The log was saved to neural_coder_workspace\\NeuralCoder${this.rand}.log  \n`
                                //     }
                                let command = "lscpu | grep 'Model name'";
                                let get_hardware = `import subprocess\nsubp = subprocess.Popen("${command}",shell=True,stdout=subprocess.PIPE,stderr=subprocess.PIPE,encoding="utf-8")\nsubp.wait(2)\nhardware = subp.communicate()[0].replace("Model name:","").strip()`;
                                let expr_hardware = { hardware: "hardware" };
                                let hard_res = NotebookUtilities.sendKernelRequestFromNotebook(this.panel, get_hardware, expr_hardware, false);
                                hard_res.then(value => {
                                    let hard = Object.values(value.hardware.data)[0];
                                    if (this.markdown) {
                                        this.markdown.model.value.text += `[NeuralCoder INFO] HardWare: ${hard}  \n`;
                                        this.markdown.model.value.text += `[NeuralCoder INFO] The log was saved to neural_coder_workspace\\NeuralCoder${this.rand}.log  \n`;
                                    }
                                    // cell.outputArea.node.innerText += `[NeuralCoder INFO] HardWare: ${hard}\n`
                                });
                                //  cell.outputArea.node.innerText += `[NeuralCoder INFO] The log was saved to neural_coder_workspace\\NeuralCoder${this.rand}.log\n`
                                const run_svg = document.createElement("svg");
                                run_svg.innerHTML = Constants.ICON_RUN;
                                (_d = (_c = (_b = (_a = run === null || run === void 0 ? void 0 : run.node.firstChild) === null || _a === void 0 ? void 0 : _a.firstChild) === null || _b === void 0 ? void 0 : _b.firstChild) === null || _c === void 0 ? void 0 : _c.firstChild) === null || _d === void 0 ? void 0 : _d.replaceWith(run_svg);
                            });
                        }
                    });
                });
            }
        }
    }
}
export class JupyterlabNotebookCodeOptimizer extends JupyterlabCodeOptimizer {
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
            NotebookActions.insertBelow(notebook);
            this.notebookTracker.currentWidget.content.activeCellIndex = count + 1;
            NotebookActions.changeCellType(notebook, 'markdown');
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
            NotebookUtilities.sendKernelRequestFromNotebook(this.panel, runcode, expr, false);
            let runcode2 = `with open('${this.log_path}', 'a' ) as f:\n       f.write("[NeuralCoder INFO] Code: User code from Jupyter Lab notebook '${this.notebookname}'\\n")`;
            let expr2 = { path: "" };
            NotebookUtilities.sendKernelRequestFromNotebook(this.panel, runcode2, expr2, false);
            let runcode3 = `with open('${this.log_path}', 'a' ) as f:\n       f.write("[NeuralCoder INFO] Benchmark Mode: Throughput\\n")`;
            let expr3 = { path: "" };
            NotebookUtilities.sendKernelRequestFromNotebook(this.panel, runcode3, expr3, false);
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
