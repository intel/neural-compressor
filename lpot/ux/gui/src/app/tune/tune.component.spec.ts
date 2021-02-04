import { async, ComponentFixture, TestBed } from '@angular/core/testing';

import { TuneComponent } from './tune.component';

describe('TuneComponent', () => {
  let component: TuneComponent;
  let fixture: ComponentFixture<TuneComponent>;

  beforeEach(async(() => {
    TestBed.configureTestingModule({
      declarations: [ TuneComponent ]
    })
    .compileComponents();
  }));

  beforeEach(() => {
    fixture = TestBed.createComponent(TuneComponent);
    component = fixture.componentInstance;
    fixture.detectChanges();
  });

  it('should create', () => {
    expect(component).toBeTruthy();
  });
});
