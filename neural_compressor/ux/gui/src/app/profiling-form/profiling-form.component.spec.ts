import { ComponentFixture, TestBed } from '@angular/core/testing';

import { ProfilingFormComponent } from './profiling-form.component';

describe('ProfilingFormComponent', () => {
  let component: ProfilingFormComponent;
  let fixture: ComponentFixture<ProfilingFormComponent>;

  beforeEach(async () => {
    await TestBed.configureTestingModule({
      declarations: [ ProfilingFormComponent ]
    })
    .compileComponents();
  });

  beforeEach(() => {
    fixture = TestBed.createComponent(ProfilingFormComponent);
    component = fixture.componentInstance;
    fixture.detectChanges();
  });

  it('should create', () => {
    expect(component).toBeTruthy();
  });
});
