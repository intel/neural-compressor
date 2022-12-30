import { PythonShell } from "python-shell";
import { MyTreeData } from "./sideBar";
import * as vscode from "vscode";
import * as DirPath from "path";
import * as fs from "fs-extra";

class CodeOptimizer {
  protected working: boolean;
  protected pathExist: boolean;
  public outPutLogPath: string;
  public outPutLogFilePath: string;
  public autoSaveLogPath: string;
  public enableSaveLogPath: string;
  public autoSaveFinalLogPath: string;
  public enableSaveFinalLogPath: string;
  public fpsList: number[];
  public treeProvider: string;
  public curCondaName?: string;
  public curPyPath: string;
  public outPutStr: string[];

  constructor() {
    this.working = false;
    this.pathExist = false;
    this.autoSaveLogPath = "../neural_coder_workspace/Auto/";
    this.enableSaveLogPath = "../neural_coder_workspace/Enable/";
    this.outPutLogPath = "../neural_coder_workspace/outPutLog/";
    this.autoSaveFinalLogPath = "";
    this.enableSaveFinalLogPath = "";
    this.outPutLogFilePath = "";

    this.fpsList = [];
    this.outPutStr = [];

    this.treeProvider = "";
    this.curCondaName = vscode.workspace
      .getConfiguration()
      .get("neuralCoder.condaName");
    this.curPyPath = "";
  }

  public registerTreeDataProvider(treeName: string, logPath: string) {
    vscode.window.registerTreeDataProvider(
      treeName,
      new MyTreeData(DirPath.resolve(logPath, "../"))
    );
  }

  // output content
  public outPutFunc(
    outPut: vscode.OutputChannel,
    outPutStr: string[],
    content: string[]
  ) {
    content.forEach((val) => {
      outPut.appendLine(val);
      outPutStr.push(val);
    });
  }

  // save log in a file
  public saveLogFile(
    outDir: string,
    logContent: string,
  ) {
    let nowTime = new Date(Date.parse(new Date().toString()));
    let nowTimeStr =
      nowTime.getFullYear() +
      "_" +
      (nowTime.getMonth() > 8 ? (nowTime.getMonth() + 1) : ('0' + (nowTime.getMonth() + 1))) + 
      "_" +
      (nowTime.getDate() > 9 ? nowTime.getDate() : ('0' + nowTime.getDate())) +
      "_" +
      (nowTime.getHours() > 9 ? nowTime.getHours() : ('0' + nowTime.getHours())) +
      "_" +
      (nowTime.getMinutes() > 9 ? nowTime.getMinutes() : ('0' + nowTime.getMinutes())) +
      "_" +
      (nowTime.getSeconds() > 9 ? nowTime.getSeconds() : ('0' + nowTime.getSeconds()));
      
    let finalPath = DirPath.resolve(`${__dirname}`, outDir + nowTimeStr);

    if (logContent === "Auto") {      
      this.autoSaveFinalLogPath = finalPath;
    } else if (logContent === "Enable") {
      this.enableSaveFinalLogPath = finalPath;
    } else if (logContent === "output")  {
      this.outPutLogFilePath = finalPath;
    }
    // mkdir file
    let isOutExist: boolean = fs.pathExistsSync(finalPath);
    if (!isOutExist) {
      fs.mkdirsSync(finalPath);
    }
  }

    // pythonShell Script
    async ncPyScript(
      dirPath: string,
      currentFilePath: string,
      feature: string,
      currentFileArgs: string,
      status: string,
      currentPythonPath: string,
      saveLogPath: string
    ) {
      return new Promise((resolve, reject) => {
        PythonShell.run(
          "NcProcessScript.py",
          {
            mode: "text",
            pythonOptions: ["-u"],
            scriptPath: dirPath,
            pythonPath: currentPythonPath,
            args: [
              currentFilePath,
              feature,
              currentFileArgs,
              status,
              saveLogPath + "/" + feature,
              currentPythonPath
            ],
          },
          (err, result) => {            
            this.pathExist = true;
            if (err) {
              // vscode.window.showErrorMessage("Please install correct package!");
              this.working = false;
            }
            resolve(result);
          }
        );
      });
    }

  // neural coder params
  async ncProcess(
    ncPath: string | undefined,
    feature: string,
    ncArgs: string | undefined,
    status: string,
    currentPythonPath: string,
    saveLogPath: string
  ) {
    let pythonRes: any;
    // find currentFile path
    const dirPath = DirPath.resolve(`${__dirname}`, "../src");
    // find running file path
    let currentFilePath = ncPath ? ncPath : "";
    let currentFileArgs = ncArgs ? ncArgs : "";

    // try {
    // asyn -> sync
    pythonRes = await this.ncPyScript(
      dirPath,
      currentFilePath,
      feature,
      currentFileArgs,
      status,
      currentPythonPath,
      saveLogPath
    );

    if (!this.pathExist) {
      vscode.window.showErrorMessage("Please input correct python Path!");
      this.working = false;
    }    
    return pythonRes;
  }

  async optimizeCode(
    feature: string,
    name: string,
    next: string,
    opt: string | undefined,
    args: string | undefined,
    ncPath: string | undefined,
    currentPythonPath: string,
    opc?: vscode.OutputChannel
  ) {
    if (opt === "normal") {
      await this.ncProcess(
        ncPath,
        feature,
        args,
        "normal",
        currentPythonPath,
        this.enableSaveFinalLogPath
      );
      this.registerTreeDataProvider(
        "Enable_Log_File",
        this.enableSaveFinalLogPath
      );
    } else {
      if (opc) {
        if (feature === "") {
          this.outPutFunc(opc, this.outPutStr, [
            `[NeuralCoder INFO] Enabling and Benchmarking for The Original Model ......`,
          ]);

          // Get the fps
          const resFps = await this.ncProcess(
            ncPath,
            "",
            args,
            "genLog",
            currentPythonPath,
            this.autoSaveFinalLogPath
          );          
          const currentFps = resFps.pop();

          this.fpsList.push(parseFloat(currentFps));

          this.outPutFunc(opc, this.outPutStr, [
            `[NeuralCoder INFO] Benchmark Result (Performance) of The Original Model is ${currentFps} (samples/second)`,
            `[NeuralCoder INFO] Enabling and Benchmarking for The Original Model ......`,
            `[NeuralCoder INFO] Enabling and Benchmarking for ${next} ......`,
          ]);
        } else {
          const resFps = await this.ncProcess(
            ncPath,
            feature,
            args,
            "genLog",
            currentPythonPath,
            this.autoSaveFinalLogPath
          );
          const currentFps = resFps.pop();
          this.fpsList.push(parseFloat(currentFps));
          this.outPutFunc(opc, this.outPutStr, [
            `[NeuralCoder INFO] Benchmark Result (Performance) of ${name} is ${currentFps} (samples/second)`,
          ]);
          if (next !== "") {
            this.outPutFunc(opc, this.outPutStr, [
              `[NeuralCoder INFO] Enabling and Benchmarking for ${next} ......`,
            ]);
          }
          if (feature === "pytorch_inc_bf16") {
            let features = [
              "",
              "pytorch_inc_static_quant_fx",
              "pytorch_inc_dynamic_quant",
              "pytorch_inc_bf16",
            ];
            let featureName = [
              "Original Model",
              "INC Enable INT8 (Static)",
              "INC Enable INT8 (Dynamic)",
              "INC Enable BF16",
            ];

            let bestFps = Math.max(...this.fpsList);
            let bestIndex = this.fpsList.indexOf(bestFps);
            let bestFeature = features[bestIndex];
            let bestFeatureName = featureName[bestIndex];
            let boost = (bestFps / this.fpsList[0]).toFixed(2);

            // Best result
            await this.ncProcess(
              ncPath,
              bestFeature,
              args,
              "normal",
              currentPythonPath,
              this.autoSaveFinalLogPath
            );
            this.outPutFunc(opc, this.outPutStr, [
              `[NeuralCoder INFO] The Best Intel Optimization: ${bestFeatureName}.`,
              `[NeuralCoder INFO] You can get up to ${boost}X performance boost.`,
            ]);

            const resHardWare = await this.ncProcess(
              ncPath,
              bestFeature,
              args,
              "hardWare",
              currentPythonPath,
              this.autoSaveFinalLogPath
            );

            // log File
            let logContent = [...this.outPutStr];
            this.saveLogFile(this.outPutLogPath, 'output');
            
            // save log file
            let outPutFinalPath = this.outPutLogFilePath + "/outPut.log";
            fs.writeFile(outPutFinalPath, logContent.join("\n"));

            this.outPutFunc(opc, this.outPutStr, [
              `[NeuralCoder INFO] HardWare: ${resHardWare}.`,
              `[NeuralCoder INFO] The log was saved to:`,
              `[NeuralCoder INFO] ${outPutFinalPath}`,
            ]);
            // TreeData
            this.registerTreeDataProvider(
              "Auto_Log_File",
              this.autoSaveFinalLogPath
            );
          }
        }
      }
    }
  }
}

export class NeuralCodeOptimizer extends CodeOptimizer {
  constructor() {
    super();
  }

  public async optimizeCodes(
    currentPythonPath: string,
    feature?: string,
    path?: string,
    args?: string,
    opc?: vscode.OutputChannel
  ) {
    if (this.working) {
      vscode.window.showInformationMessage("Not done yet");
      return;
    }

    this.working = true;
    const optimizeType =
      feature !== undefined ? feature : "pytorch_mixed_precision_cpu";
    if (feature === "auto-quant") {
      // outPut init
      if (opc) {
        opc.clear();
        opc.show();

        this.outPutFunc(opc, this.outPutStr, [
          "[NeuralCoder INFO] Auto-Quant Started ......",
        ]);

        // mkdir autoSaveLogs
        this.saveLogFile(this.autoSaveLogPath, "Auto");

        this.fpsList = [];
        let pathFinal = path ? path : "";
        const currentFileName = pathFinal.split(/[\\\/]/).pop();

        this.outPutFunc(opc, this.outPutStr, [
          `[NeuralCoder INFO] Code: User code from VS Code "${currentFileName}"`,
          `[NeuralCoder INFO] Benchmark Mode: Throughput`,
        ]);

        await this.optimizeCode(
          "", //current feature
          "The Original Model", //current feature name
          "INC Enable INT8 (Static)", //next feature name
          "auto", //normal or auto
          args, //parameters
          path,
          currentPythonPath,
          opc
        );
        await this.optimizeCode(
          "pytorch_inc_static_quant_fx",
          "INC Enable INT8 (Static)",
          "INC Enable INT8 (Dynamic)",
          "auto",
          args,
          path,
          currentPythonPath,
          opc
        );
        await this.optimizeCode(
          "pytorch_inc_dynamic_quant",
          "INC Enable INT8 (Dynamic)",
          "INC Enable BF16",
          "auto",
          args,
          path,
          currentPythonPath,
          opc
        );
        await this.optimizeCode(
          "pytorch_inc_bf16",
          "INC Enable BF16",
          "",
          "auto",
          args,
          path,
          currentPythonPath,
          opc
        );
      }
    } else {
      this.saveLogFile(this.enableSaveLogPath, "Enable");
      await this.optimizeCode(
        optimizeType,
        "",
        "",
        "normal",
        args,
        path,
        currentPythonPath
      );
    }
    this.working = false;
  }
}
