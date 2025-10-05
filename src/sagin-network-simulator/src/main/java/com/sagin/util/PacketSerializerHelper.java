package com.sagin.util; 

import com.sagin.model.Packet;
import com.fasterxml.jackson.core.JsonProcessingException;
import com.fasterxml.jackson.databind.ObjectMapper;

/**
 * Lớp trợ giúp (Helper) để xử lý việc chuyển đổi giữa đối tượng Packet và định dạng JSON String.
 * Sử dụng ObjectMapper của Jackson.
 */
public class PacketSerializerHelper {

    // Sử dụng một instance duy nhất (singleton) của ObjectMapper. 
    // ObjectMapper là thread-safe, nên việc dùng static final là an toàn và hiệu quả.
    private static final ObjectMapper OBJECT_MAPPER = new ObjectMapper();

    /**
     * Chuyển đổi đối tượng Packet thành chuỗi JSON String để gửi qua mạng.
     * * @param packet Đối tượng Packet cần tuần tự hóa.
     * @return Chuỗi JSON String, hoặc null nếu xảy ra lỗi.
     */
    public static String serialize(Packet packet) {
        if (packet == null) {
            return null;
        }
        try {
            // ObjectMapper sẽ tôn trọng các annotation trong lớp Packet (như @JsonInclude)
            return OBJECT_MAPPER.writeValueAsString(packet);
        } catch (JsonProcessingException e) {
            System.err.println("Lỗi tuần tự hóa (Serialization) Packet: " + e.getMessage());
            // Tùy chọn: ném RuntimeException hoặc trả về null
            return null; 
        }
    }

    /**
     * Chuyển đổi chuỗi JSON String nhận được qua mạng thành đối tượng Packet.
     * * @param jsonString Chuỗi JSON cần giải tuần tự hóa.
     * @return Đối tượng Packet, hoặc null nếu xảy ra lỗi.
     */
    public static Packet deserialize(String jsonString) {
        if (jsonString == null || jsonString.isEmpty()) {
            return null;
        }
        try {
            // ObjectMapper sẽ tôn trọng các annotation trong lớp Packet (như @JsonIgnoreProperties)
            return OBJECT_MAPPER.readValue(jsonString, Packet.class);
        } catch (JsonProcessingException e) {
            System.err.println("Lỗi giải tuần tự hóa (Deserialization) JSON: " + e.getMessage());
            // Tùy chọn: ném RuntimeException hoặc trả về null
            return null; 
        }
    }
}