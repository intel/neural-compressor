import { TreeDataProvider, TreeItem, TreeItemCollapsibleState, ProviderResult, window } from "vscode";
import * as  fs from "fs";
import * as path from "path";

export class MyTreeData implements TreeDataProvider<MyTreeItem>{
    constructor(private rootPath: string){
    }

    getTreeItem(element: MyTreeItem) : MyTreeItem | Thenable<MyTreeItem> {
        return element;
    }

    getChildren(element?: MyTreeItem | undefined): ProviderResult<MyTreeItem[]>{
        if(!this.rootPath){
            window.showInformationMessage('No file in empty directory');
            return Promise.resolve([]);
        }
        if(element === undefined){
            return Promise.resolve(this.searchFiles(this.rootPath));
        }
        else{
            return Promise.resolve(this.searchFiles(path.join(element.parentPath, element.label)));
        }
    }
    //search file
    private searchFiles(parentPath: string): MyTreeItem[] {
        var treeDir: MyTreeItem[] = [];
        if(this.pathExists(parentPath)){
            var fsReadDir = fs.readdirSync(parentPath, 'utf-8');
            fsReadDir.forEach(fileName => {
                var filePath = path.join(parentPath, fileName);//absolute Path
                if(fs.statSync(filePath).isDirectory()){//Directory
                    treeDir.push(new MyTreeItem(fileName, parentPath, TreeItemCollapsibleState.Collapsed));
                }
                else{//file
                    treeDir.push(new MyTreeItem(fileName, parentPath, TreeItemCollapsibleState.None));
                }
            });
        }
        return treeDir;
    }   
    //pathExists
    private pathExists(filePath: string): boolean{
        try{
            fs.accessSync(filePath);
        }
        catch(err){
            return false;
        }
        return true;
    }
}

export class MyTreeItem extends TreeItem{
    constructor(
        public readonly label: string,      //save current label
        public readonly parentPath: string,   //save current label Path
        public readonly collapsibleState: TreeItemCollapsibleState
    ){
        super(label, collapsibleState);
    }
    //click method
    command = {
        title: "this.label",
        command: 'MyTreeItem.itemClick',
        arguments: [    //params
            this.label,
            path.join(this.parentPath, this.label)
        ]
    };
    contextValue = 'MyTreeItem';//provide for when
}
