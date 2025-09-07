package com.sagsins.core.model;

import lombok.*;

@Getter
@Setter
@AllArgsConstructor
@NoArgsConstructor
public class Velocity {
    private double vx; // vận tốc trục X (m/s)
    private double vy; // vận tốc trục Y (m/s)
    private double vz; // vận tốc trục Z (m/s)
}
