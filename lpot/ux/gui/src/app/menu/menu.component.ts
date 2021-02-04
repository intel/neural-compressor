import { Component, OnInit } from '@angular/core';
import { ModelService } from '../services/model.service';


@Component({
  selector: 'app-menu',
  templateUrl: './menu.component.html',
  styleUrls: ['./menu.component.scss']
})
export class MenuComponent implements OnInit {

  workspacePath: string;
  editable = false;
  constructor(
    private modelService: ModelService
  ) { }

  ngOnInit() {
    this.modelService.getDefaultPath('workspace')
      .subscribe(repoPath => {
        this.workspacePath = repoPath['path'];
        this.modelService.workspacePath = repoPath['path'];
      })
  }

  saveWorkspace(event?) {
    if ((event && event.key === 'Enter') || event === 'save') {
      this.editable = false;
      this.modelService.setWorkspacePath(this.workspacePath)
        .subscribe(resp => console.log(resp));
    }
  }

  openUrl(url: string) {
    window.open(url);
  }

}
