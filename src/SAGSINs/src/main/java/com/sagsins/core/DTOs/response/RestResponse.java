package com.sagsins.core.DTOs.response;

import lombok.AllArgsConstructor;
import lombok.Getter;
import lombok.NoArgsConstructor;
import lombok.Setter;

@Getter
@Setter
@AllArgsConstructor
@NoArgsConstructor
public class RestResponse<T>{
    private int status;
    private String error;
    private Object message;
    private T data;

    public RestResponse(int status, String error, Object message) {
        this.status = status;
        this.error = error;
        this.message = message;
    }
}