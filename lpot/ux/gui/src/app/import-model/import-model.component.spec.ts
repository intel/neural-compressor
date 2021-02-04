import { async, ComponentFixture, TestBed } from '@angular/core/testing';

import { ImportModelComponent } from './import-model.component';

describe('ImportModelComponent', () => {
  let component: ImportModelComponent;
  let fixture: ComponentFixture<ImportModelComponent>;

  beforeEach(async(() => {
    TestBed.configureTestingModule({
      declarations: [ImportModelComponent]
    })
      .compileComponents();
  }));

  beforeEach(() => {
    fixture = TestBed.createComponent(ImportModelComponent);
    component = fixture.componentInstance;
    fixture.detectChanges();
  });

  it('should create', () => {
    expect(component).toBeTruthy();
  });
});
