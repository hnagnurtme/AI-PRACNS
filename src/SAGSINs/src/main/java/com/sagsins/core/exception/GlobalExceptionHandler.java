package com.sagsins.core.exception;

import java.util.Set;
import java.util.stream.Collectors;
import java.util.Arrays;
import java.util.HashSet;
import java.util.List;

import org.springframework.http.HttpStatus;
import org.springframework.http.ResponseEntity;
import org.springframework.validation.BindingResult;
import org.springframework.validation.FieldError;
import org.springframework.web.bind.MethodArgumentNotValidException;
import org.springframework.web.bind.annotation.ExceptionHandler;
import org.springframework.web.bind.annotation.RestControllerAdvice;
import org.springframework.web.multipart.MaxUploadSizeExceededException;

import com.sagsins.core.DTOs.response.Message;
import com.sagsins.core.DTOs.response.RestResponse;



@RestControllerAdvice
public class GlobalExceptionHandler {
    private static final Set<ExceptionConstants> UNPROCESSABLE_ENTITY_EXCEPTIONS = new HashSet<>(Arrays.asList(
        ExceptionConstants.DUPLICATE_KEY
    ));

    @ExceptionHandler(DuplicateKeyException.class)
    public ResponseEntity<RestResponse<Object>> handleNotMatchPasswordException(DuplicateKeyException e) {
        return buildResponse(ExceptionConstants.DUPLICATE_KEY, e.getMessage());
    }
    
    @ExceptionHandler(NotFoundException.class)
    public ResponseEntity<RestResponse<Object>> handleNotFoundException(NotFoundException e) {
        return buildResponse(ExceptionConstants.NOT_FOUND_EXCEPTION, e.getMessage());
    }
    
    @ExceptionHandler(MethodArgumentNotValidException.class)
    public ResponseEntity<RestResponse<Object>> handleMethodArgumentNotValidException(MethodArgumentNotValidException e) {
        BindingResult result = e.getBindingResult();
        List<FieldError> fieldErrors = result.getFieldErrors();
        List<String> errors = fieldErrors.stream().map(FieldError::getDefaultMessage).collect(Collectors.toList());

        RestResponse<Object> response = new RestResponse<>();
        response.setStatus(HttpStatus.BAD_REQUEST.value());
        response.setError("Validation Error");
        response.setMessage(errors.size() > 1 ? errors.toString() : errors.get(0));

        return ResponseEntity.status(HttpStatus.UNPROCESSABLE_ENTITY).body(response);
    }

    @ExceptionHandler(Exception.class)
    public ResponseEntity<RestResponse<Object>> handleGenericException(Exception e) {
        return buildResponse(ExceptionConstants.INTERNAL_SERVER_ERROR,e.getMessage());
    }

    @ExceptionHandler(MaxUploadSizeExceededException.class)
    public ResponseEntity<Message> handleMaxSizeException(MaxUploadSizeExceededException exc) {
        return ResponseEntity.status(HttpStatus.EXPECTATION_FAILED)
                            .body(new Message("File quá lớn!"));
    }

    private ResponseEntity<RestResponse<Object>> buildResponse(ExceptionConstants error, String message) {
        RestResponse<Object> response = new RestResponse<>();
        response.setStatus(error.getCode());
        response.setError(error.getMessageName());
        response.setMessage(message);
        
        HttpStatus httpStatus = UNPROCESSABLE_ENTITY_EXCEPTIONS.contains(error) 
                ? HttpStatus.UNPROCESSABLE_ENTITY 
                : HttpStatus.BAD_REQUEST;
        
        return ResponseEntity.status(httpStatus).body(response);
    }
}