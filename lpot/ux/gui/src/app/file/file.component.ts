import { Component, Input, OnInit } from '@angular/core';
import { ModelService } from '../services/model.service';

@Component({
  selector: 'app-file',
  templateUrl: './file.component.html',
  styleUrls: ['./file.component.scss']
})
export class FileComponent implements OnInit {

  @Input() id: string;
  @Input() fileType: string;
  fileText = '';

  constructor(
    private modelService: ModelService
  ) { }

  ngOnInit() {
    this.getFile();
  }

  getFile() {
    this.modelService.getFile(this.id, this.fileType).subscribe(data => {
      this.fileText = String(data);
    });
  }

}
