import { Component, OnInit } from '@angular/core';
import { MatDialog } from '@angular/material';
import { ErrorComponent } from '../error/error.component';
import { FileBrowserComponent } from '../file-browser/file-browser.component';
import { ModelService } from '../services/model.service';


@Component({
  selector: 'app-menu',
  templateUrl: './menu.component.html',
  styleUrls: ['./menu.component.scss']
})
export class MenuComponent implements OnInit {

  workspacePath: string;

  constructor(
    private modelService: ModelService,
    public dialog: MatDialog
  ) { }

  ngOnInit() {
    this.modelService.getDefaultPath('workspace')
      .subscribe(repoPath => {
        this.workspacePath = repoPath['path'];
        this.modelService.workspacePath = repoPath['path'];
      })
  }

  openUrl(url: string) {
    window.open(url);
  }

  openDialog() {
    const dialogRef = this.dialog.open(FileBrowserComponent, {
      width: '60%',
      height: '60%',
      data: {
        path: this.modelService.workspacePath,
        files: false
      }
    });

    dialogRef.afterClosed().subscribe(chosenFile => {
      if (chosenFile) {
        this.workspacePath = chosenFile;
        this.modelService.setWorkspacePath(chosenFile)
          .subscribe(
            response => {
              this.modelService.workspacePathChange.next(true);
            },
            error => {
              this.openErrorDialog(error);
            }
          );
      }
    });;
  }

  openErrorDialog(error) {
    const dialogRef = this.dialog.open(ErrorComponent, {
      data: error
    });
  }

}
