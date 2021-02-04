import { async, ComponentFixture, TestBed } from '@angular/core/testing';

import { PredefinedModelsComponent } from './predefined-models.component';

describe('PredefinedModelsComponent', () => {
  let component: PredefinedModelsComponent;
  let fixture: ComponentFixture<PredefinedModelsComponent>;

  beforeEach(async(() => {
    TestBed.configureTestingModule({
      declarations: [ PredefinedModelsComponent ]
    })
    .compileComponents();
  }));

  beforeEach(() => {
    fixture = TestBed.createComponent(PredefinedModelsComponent);
    component = fixture.componentInstance;
    fixture.detectChanges();
  });

  it('should create', () => {
    expect(component).toBeTruthy();
  });
});
