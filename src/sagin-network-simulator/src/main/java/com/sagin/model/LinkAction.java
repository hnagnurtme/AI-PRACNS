package com.sagin.model;

import lombok.*;
import com.fasterxml.jackson.annotation.JsonInclude;

@Getter
@Setter
@AllArgsConstructor
@NoArgsConstructor
@JsonInclude(JsonInclude.Include.NON_NULL)
public class LinkAction {

    private int modulationScheme;   
    private double codingRate;      
    private double transmitPowerDbm; 
    private int schedulingPriority;  
}