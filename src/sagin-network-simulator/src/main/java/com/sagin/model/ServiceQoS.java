package com.sagin.model;

import lombok.AllArgsConstructor;
import lombok.Getter;



/**
 * Định nghĩa yêu cầu QoS mặc định cho từng loại dịch vụ.
 */
@Getter
@AllArgsConstructor
public class ServiceQoS {

    private String serviceType;
    private int defaultPriority;
    private double maxLatencyMs;
    private double maxLossRate;
}