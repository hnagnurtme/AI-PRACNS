package com.sagsins.core.model;

import lombok.*;

@Getter
@Setter
@AllArgsConstructor
@NoArgsConstructor
public class Orbit {
    private String type;         // LEO, MEO, GEO
    private double inclination;  // độ nghiêng
    private double period;       // chu kỳ quỹ đạo (phút)
    private double semiMajorAxis; // bán trục lớn (km)
}
