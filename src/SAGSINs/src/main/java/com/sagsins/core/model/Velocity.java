package com.sagsins.core.model;

import lombok.*;

import com.fasterxml.jackson.annotation.JsonIgnore;
import com.fasterxml.jackson.annotation.JsonIgnoreProperties;
import com.fasterxml.jackson.annotation.JsonInclude;
import com.fasterxml.jackson.annotation.JsonProperty;
import com.fasterxml.jackson.annotation.JsonProperty.Access;

@Getter
@Setter
@AllArgsConstructor
@NoArgsConstructor
@JsonInclude(JsonInclude.Include.NON_NULL)
@ToString
@JsonIgnoreProperties(value = {"speed", "moving"}, ignoreUnknown = true)
public class Velocity {

    // Vận tốc theo trục X (Đông/Tây) - Ví dụ: tính bằng km/s
    private double velocityX; 
    
    // Vận tốc theo trục Y (Bắc/Nam) - Ví dụ: tính bằng km/s
    private double velocityY; 
    
    // Vận tốc theo trục Z (Lên/Xuống) - Ví dụ: tính bằng km/s
    private double velocityZ; 

    /**
     * Tính độ lớn (tốc độ) tổng thể của vector vận tốc.
     * Công thức: speed = sqrt(vx^2 + vy^2 + vz^2)
     * @return Tốc độ tổng thể (speed)
     */
    @JsonIgnore
    @JsonProperty(access = Access.READ_ONLY)
    public double getSpeed() {
        return Math.sqrt(velocityX * velocityX + velocityY * velocityY + velocityZ * velocityZ);
    }
    
    /**
     * Kiểm tra xem Node có đang di chuyển đáng kể không.
     * @return true nếu tốc độ lớn hơn 0.001 km/s.
     */
    @JsonIgnore
    public boolean isMoving() {
        return getSpeed() > 1e-3; 
    }
}