package com.sagin.util; 

import com.sagin.model.Packet;
import com.fasterxml.jackson.core.JsonProcessingException;
import com.fasterxml.jackson.databind.ObjectMapper;
import com.fasterxml.jackson.databind.SerializationFeature;
import com.fasterxml.jackson.databind.DeserializationFeature;

import java.util.List;

import org.slf4j.Logger;
import org.slf4j.LoggerFactory;

/**
 * Lớp trợ giúp (Helper) để xử lý việc chuyển đổi giữa đối tượng Packet và định dạng JSON String.
 * Sử dụng ObjectMapper của Jackson.
 */
public class PacketSerializerHelper {

    private static final Logger logger = LoggerFactory.getLogger(PacketSerializerHelper.class);
    
    // Cấu hình ObjectMapper để có thể đọc/ghi các thuộc tính của Lombok và Enums
    private static final ObjectMapper OBJECT_MAPPER = new ObjectMapper()
            .configure(SerializationFeature.FAIL_ON_EMPTY_BEANS, false) // Tránh lỗi nếu Packet trống
            .configure(DeserializationFeature.READ_ENUMS_USING_TO_STRING, true) // Đọc enum từ chuỗi
            .configure(DeserializationFeature.FAIL_ON_UNKNOWN_PROPERTIES, false); // Bỏ qua các thuộc tính không xác định

    /**
     * Chuyển đổi đối tượng Packet thành chuỗi JSON String để gửi qua mạng.
     * @param packet Đối tượng Packet cần tuần tự hóa.
     * @return Chuỗi JSON String, hoặc null nếu xảy ra lỗi.
     */
    public static String serialize(Packet packet) {
        if (packet == null) {
            return null;
        }
        try {
            return OBJECT_MAPPER.writeValueAsString(packet);
        } catch (JsonProcessingException e) {
            logger.error("Lỗi tuần tự hóa (Serialization) Packet {}: {}", packet.getPacketId(), e.getMessage());
            return null; 
        }
    }

    /**
     * Chuyển đổi chuỗi JSON String nhận được qua mạng thành đối tượng Packet.
     * @param jsonString Chuỗi JSON cần giải tuần tự hóa.
     * @return Đối tượng Packet, hoặc null nếu xảy ra lỗi.
     */
    public static Packet deserialize(String jsonString) {
        if (jsonString == null || jsonString.isEmpty()) {
            logger.warn("JSON string is null or empty");
            return null;
        }
        
        try {
            if (jsonString.trim().startsWith("[")) {
                // Trường hợp 1: Client gửi mảng [{...}]
                // Đọc mảng và lấy phần tử đầu tiên (hoặc xử lý lỗi nếu mảng rỗng)
                List<Packet> packets = OBJECT_MAPPER.readValue(jsonString, new com.fasterxml.jackson.core.type.TypeReference<List<Packet>>() {});
                if (packets.isEmpty()) {
                    logger.warn("JSON array is empty");
                    return null;
                }
                logger.info("Successfully deserialized packet from JSON array: {}", packets.get(0).getPacketId());
                return packets.get(0);
            } else {
                // Trường hợp 2: Client gửi đối tượng {...}
                Packet packet = OBJECT_MAPPER.readValue(jsonString, Packet.class);
                logger.info("Successfully deserialized packet from JSON object: {}", packet.getPacketId());
                return packet;
            }
        } catch (JsonProcessingException e) {
            logger.error("Lỗi giải tuần tự hóa (Deserialization) JSON: {}. JSON string: [{}]", e.getMessage(), jsonString);
            logger.error("Chi tiết lỗi: ", e);
            return null; 
        }
    }
}