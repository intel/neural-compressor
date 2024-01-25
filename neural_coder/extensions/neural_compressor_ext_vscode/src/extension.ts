import * as vscode from "vscode";
import { NeuralCodeOptimizer } from "./neuralcoder";
import * as DirPath from "path";

// highLight
async function highLight () {
  const editor = vscode.window.activeTextEditor;
  if (!editor) {
    return;
  }
  const document = editor.document;
  
  const text = document.getText();
  const regStart = /# \[NeuralCoder\] .*? \[Beginning Line\]/g;
  const regEnd = /# \[NeuralCoder\] .*? \[Ending Line\]/g;
  const startMatch = regStart.exec(text);
  const endMatch = regEnd.exec(text);
  if(startMatch && endMatch) {
    const startLine = document.positionAt(startMatch.index);
    const endLine = document.positionAt(endMatch.index);
    const start = document.lineAt(startLine).range.start.character;
    const end = document.lineAt(endLine).range.end.character;
    const range: vscode.Range = new vscode.Range(Number(startLine), start, Number(endLine), end);
    // highLight
    let path = vscode.window.activeTextEditor?.document.fileName;
    if (path) {
      let filePath = DirPath.resolve(`${__dirname}`, "../", path);
      // vscode.workspace.openTextDocument(filePath).then(async (document) => {
      //    await vscode.window.showTextDocument(document , {preserveFocus: false, selection: range, viewColumn: vscode.ViewColumn.One});
      // });
    }    
    }
}

export async function activate(context: vscode.ExtensionContext) {  
  // init
  const pythonPath = "neuralCoder.pythonPath";
  let config: vscode.WorkspaceConfiguration =
    vscode.workspace.getConfiguration();
  let currentCondaName = config.get<string>(pythonPath);

  if (!currentCondaName) {
    vscode.window.showErrorMessage("Please input python Path!");
    return;
  }
  // conda Env
  context.subscriptions.push(
    vscode.workspace.onDidChangeConfiguration(() => {
      const currentCondaName = vscode.workspace.getConfiguration().get(pythonPath);
      if (!currentCondaName) {
        vscode.window.showErrorMessage("Please input python Path!");
        return;
      }
    })
  );

  // start
  let path = vscode.window.activeTextEditor?.document.fileName;  
  let userInput: string = "";
  let ncCoder = new NeuralCodeOptimizer();
  let curPythonPath = currentCondaName;

  let incEnableINT8Static = vscode.commands.registerCommand(
    "neuralCoder.incEnableINT8Static",
    () => {
      vscode.window.withProgress(
        {
          cancellable: false,
          location: vscode.ProgressLocation.Notification,
          title: "Running INT8 Static!",
        },
        async () => {          
          ncCoder.optimizeCodes(
            curPythonPath,
            "pytorch_inc_static_quant_fx",
            path,
            ""
          );
        }
      );
    }
  );
  let incEnableINT8Dynamic = vscode.commands.registerCommand(
    "neuralCoder.incEnableINT8Dynamic",
    () => {
      vscode.window.withProgress(
        {
          cancellable: false,
          location: vscode.ProgressLocation.Notification,
          title: "Running INT8 Dynamic!",
        },
        async () => {
          ncCoder.optimizeCodes(
            curPythonPath,
            "pytorch_inc_dynamic_quant",
            path,
            ""
          );
        }
      );
    }
  );
  let incEnableBF16 = vscode.commands.registerCommand(
    "neuralCoder.incEnableBF16",
    () => {
      vscode.window.withProgress(
        {
          cancellable: false,
          location: vscode.ProgressLocation.Notification,
          title: "Running BF16!",
        },
        async () => {
          ncCoder.optimizeCodes(curPythonPath, "pytorch_inc_bf16", path, "");
        }
      );
    }
  );

  let incAutoEnableBenchmark = vscode.commands.registerCommand(
    "neuralCoder.incAutoEnableBenchmark",
    () => {
      vscode.window.withProgress(
        {
          cancellable: false,
          location: vscode.ProgressLocation.Notification,
          title: "Running AutoEnableBenchmark!",
        },
        async () => {
          vscode.window
            .showInputBox({
              password: false, // need password?
              ignoreFocusOut: true, // when focus other thing
              placeHolder: "INPUT EXECUTE PARAMETERS OR NOT: ", // hint
            })
            .then((value) => {
              if (typeof value !== "undefined") {
                userInput = value ? value : "";
                const opc =
                  vscode.window.createOutputChannel("Neural Coder Auto-Bench");
                ncCoder.optimizeCodes(
                  curPythonPath,
                  "auto-quant",
                  path,
                  userInput,
                  opc
                );
              }
            });
        }
      );
    }
  );

  context.subscriptions.push(incEnableINT8Static);
  context.subscriptions.push(incEnableINT8Dynamic);
  context.subscriptions.push(incEnableBF16);
  context.subscriptions.push(incAutoEnableBenchmark);
  context.subscriptions.push(vscode.workspace.onDidChangeTextDocument((e: vscode.TextDocumentChangeEvent) => {
    highLight();
}));

  //register command MyTreeItem.itemClick
  context.subscriptions.push(
    vscode.commands.registerCommand(
      "MyTreeItem.itemClick",
      (label, filePath) => {
        //display content
        vscode.workspace.openTextDocument(filePath)
        .then(doc => {
            vscode.window.showTextDocument(doc);
        }, err => {
            console.log(`Open ${filePath} error, ${err}.`);
        }).then(undefined, err => {
            console.log(`Open ${filePath} error, ${err}.`);
        });
      }
    )
  );
}

// this method is called when your extension is deactivated
export function deactivate() {}

