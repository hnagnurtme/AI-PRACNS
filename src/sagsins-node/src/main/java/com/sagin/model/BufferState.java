package com.sagin.model;


import lombok.AllArgsConstructor;
import lombok.Data;
import lombok.NoArgsConstructor;

@Data
@AllArgsConstructor
@NoArgsConstructor
public class BufferState {
    private int queueSize;
    private double bandwidthUtilization;
}
