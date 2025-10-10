package com.sagsins.core.DTOs.response;

import lombok.AllArgsConstructor;
import lombok.Getter;
import lombok.NoArgsConstructor;
import lombok.Setter;

@Getter
@Setter
@AllArgsConstructor
@NoArgsConstructor
public class DockerResposne {
    private String nodeId;
    private String pid;     
    private String status; 

    // Tiện thể tạo helper constructor cho boolean
    public DockerResposne(String nodeId, String pid, boolean isRunning) {
        this.nodeId = nodeId;
        this.pid = pid;
        this.status = isRunning ? "running" : "stopped";
    }
}
