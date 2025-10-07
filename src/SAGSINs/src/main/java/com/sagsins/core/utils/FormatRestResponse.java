package com.sagsins.core.utils;

import org.springframework.core.MethodParameter;
import org.springframework.http.MediaType;
import org.springframework.http.server.ServerHttpRequest;
import org.springframework.http.server.ServerHttpResponse;
import org.springframework.http.server.ServletServerHttpResponse;
import org.springframework.web.bind.annotation.ControllerAdvice;
import org.springframework.web.servlet.mvc.method.annotation.ResponseBodyAdvice;

import com.sagsins.core.DTOs.response.RestResponse;

import jakarta.servlet.http.HttpServletResponse;

@ControllerAdvice
public class FormatRestResponse implements ResponseBodyAdvice<Object> {

    @Override
    public boolean supports(MethodParameter returnType,  Class converterType) {
        return true;
    }

    @Override
    public Object beforeBodyWrite( Object body, MethodParameter returnType, MediaType selectedContentType,
            Class selectedConverterType,  ServerHttpRequest request,  ServerHttpResponse response) {

        HttpServletResponse httpServletResponse = ((ServletServerHttpResponse) response).getServletResponse();
        int status = httpServletResponse.getStatus();
        
        if( selectedContentType.equals(MediaType.IMAGE_JPEG) || selectedContentType.equals(MediaType.IMAGE_PNG)){
            return body;
        }

        RestResponse<Object> restResponse = new RestResponse<>();

        if( body instanceof String){
            return body;
        }
        else if( body instanceof RestResponse){
            return body;
        }

        else if( body instanceof byte[]){
            return body;
        }

        else

        if( status >= 400){
           return body;
        }

        else{
            restResponse.setStatus(status);
            restResponse.setMessage("Success");
            restResponse.setData(body);
        }

        return restResponse;
    }
    
}