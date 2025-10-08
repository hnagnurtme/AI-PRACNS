package com.sagin.model;

import lombok.*;

import com.fasterxml.jackson.annotation.JsonIgnore;
import com.fasterxml.jackson.annotation.JsonInclude;

@Getter
@Setter
@AllArgsConstructor
@NoArgsConstructor
@JsonInclude(JsonInclude.Include.NON_NULL)
@ToString
public class Orbit {
    
    // Bán trục lớn (Độ lớn quỹ đạo) - Ví dụ: tính bằng km
    private double semiMajorAxisKm; 
    
    // Độ lệch tâm (Hình dạng quỹ đạo: 0 = tròn, >0 = elip)
    private double eccentricity;    
    
    // Độ nghiêng (Góc quỹ đạo so với mặt phẳng xích đạo) - Ví dụ: tính bằng độ
    private double inclinationDeg;  
    
    // Kinh độ của nút lên (Hướng của quỹ đạo trong không gian) - Ví dụ: tính bằng độ
    private double raanDeg;         
    
    // Argument of Perigee (Hướng của quỹ đạo elip) - Ví dụ: tính bằng độ
    private double argumentOfPerigeeDeg; 
    
    // True Anomaly (Vị trí thực tế của vệ tinh trên quỹ đạo) - Ví dụ: tính bằng độ
    private double trueAnomalyDeg;  

    /** * Kiểm tra xem quỹ đạo có phải là quỹ đạo tròn hoàn hảo không (e = 0).
     */
    @JsonIgnore
    public boolean isCircular() {
        // Dùng một ngưỡng nhỏ để tính toán dấu phẩy động
        return eccentricity < 1e-6; 
    }
}