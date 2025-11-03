package com.sagin.DTOs;

import java.util.List;

import lombok.AllArgsConstructor;
import lombok.Data;
import lombok.NoArgsConstructor;

@Data
@AllArgsConstructor
@NoArgsConstructor
public class RoutingResponse {
    private String nextHopNodeId;
    private List<String> path;
    private String algorithm;
    private String status;
}
