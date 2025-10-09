package com.sagsins.core.DTOs.response;

import lombok.AllArgsConstructor;
import lombok.Data;


@Data
@AllArgsConstructor

public class HealthResposne {
    private String status;
    private String message;
}
