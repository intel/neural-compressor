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
import { EMPTY, Observable } from 'rxjs';
import { ModelService } from '../services/model.service';

@Injectable()
export class AuthenticationInterceptor implements HttpInterceptor {

  constructor(
    private modelService: ModelService
  ) { }

  getToken(): string {
    if (localStorage.getItem('token')) {
      return localStorage.getItem('token');
    } else {
      this.modelService.getToken()
        .subscribe(response => {
          this.modelService.setToken(response['token']);
          return response['token'];
        });
    }
  }

  intercept(request: HttpRequest<unknown>, next: HttpHandler): Observable<HttpEvent<unknown>> {
    let headers = new HttpHeaders({
      'Authorization': this.getToken(),
    });

    if (!localStorage.getItem('token')) {
      headers = new HttpHeaders({
        'Content-Type': 'application/vnd.api+json'
      });
    }

    const modifiedReq = request.clone({
      headers: headers
    });

    if (localStorage.getItem('token') || request.url.includes('token')) {
      return next.handle(modifiedReq);
    } else {
      return EMPTY;
    }
  }
}
