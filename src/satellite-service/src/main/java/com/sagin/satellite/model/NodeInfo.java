package com.sagin.satellite.model;
import lombok.*;

@Getter
@Setter
@AllArgsConstructor
@NoArgsConstructor
public class NodeInfo {
    private String id;
    private String type = "satellite";
    private String ip;
    private Coordinates coords;
    private boolean active;
    private double bandwidth;
    private double load;
    private long lastUpdated;
}
