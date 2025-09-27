package com.sagsins.core.DTOs;

import lombok.AllArgsConstructor;
import lombok.Data;
import lombok.NoArgsConstructor;

@Data
@AllArgsConstructor
@NoArgsConstructor
public class UpdateNodeRequest {
    private String nodeId;
    private String nodeType;
    private CreateNodeRequest.PositionDto position;
    private CreateNodeRequest.OrbitDto orbit;
    private CreateNodeRequest.VelocityDto velocity;
}
