import { Pipe, PipeTransform } from '@angular/core';

@Pipe({ name: 'modelList' })
export class ModelListPipe implements PipeTransform {
  transform(value: string): string {
    value === 'acc_float32' ? value = 'fp32 baseline' : null;
    value === 'acc_int8' ? value = 'int8 accuracy' : null;
    value = value.replace('_', ' ');
    return value;
  }
}