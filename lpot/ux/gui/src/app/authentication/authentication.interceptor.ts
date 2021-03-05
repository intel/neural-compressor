// Copyright (c) 2021 Intel Corporation
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
// http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.
import { Injectable } from '@angular/core';
import {
  HttpRequest,
  HttpHandler,
  HttpEvent,
  HttpInterceptor,
  HttpHeaders,
} from '@angular/common/http';
import { Observable } from 'rxjs';
import { ModelService } from '../services/model.service';

@Injectable()
export class AuthenticationInterceptor implements HttpInterceptor {

  token;

  constructor(
    private modelService: ModelService
  ) { }

  intercept(request: HttpRequest<unknown>, next: HttpHandler): Observable<HttpEvent<unknown>> {
    if (localStorage.getItem('token')) {
      this.token = localStorage.getItem('token');

      let headers = new HttpHeaders({
        'Authorization': this.token,
      });

      const modifiedReq = request.clone({
        headers: headers
      });

      return next.handle(modifiedReq);

    } else if (request.url.includes('token')) {

      return next.handle(request);

    } else {
      this.modelService.getToken()
        .subscribe(response => {
          this.modelService.setToken(response['token']);
          this.token = response['token'];

          let headers = new HttpHeaders({
            'Authorization': this.token,
          });

          const modifiedReq = request.clone({
            headers: headers
          });

          return next.handle(modifiedReq);
        });
    }
  }
}
