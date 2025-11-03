package com.example.util;

import com.example.model.Packet;
import com.fasterxml.jackson.core.JsonProcessingException;
import com.fasterxml.jackson.databind.ObjectMapper;
import com.fasterxml.jackson.databind.SerializationFeature;
import com.fasterxml.jackson.databind.DeserializationFeature;

import org.slf4j.Logger;
import org.slf4j.LoggerFactory;

/**
 * Lớp trợ giúp (Helper) để xử lý việc chuyển đổi giữa đối tượng Java và định
 * dạng JSON String.
 * Sử dụng ObjectMapper của Jackson để đảm bảo tính tương thích RPC/Socket.
 */
public class PacketSerializerHelper {

    private static final Logger logger = LoggerFactory.getLogger(PacketSerializerHelper.class);

    // Cấu hình ObjectMapper: thread-safe và được tái sử dụng.
    private static final ObjectMapper OBJECT_MAPPER = new ObjectMapper()
            .configure(SerializationFeature.FAIL_ON_EMPTY_BEANS, false)
            .configure(DeserializationFeature.READ_ENUMS_USING_TO_STRING, true)
            .configure(DeserializationFeature.FAIL_ON_UNKNOWN_PROPERTIES, false); // Bỏ qua thuộc tính không xác định

    /**
     * Chuyển đổi một đối tượng Java thành chuỗi JSON String để gửi qua mạng.
     *
     * @param object Đối tượng cần tuần tự hóa (ví dụ: Packet hoặc
     *               PacketTransferWrapper).
     * @return Chuỗi JSON String, hoặc null nếu xảy ra lỗi.
     */
    public static String serialize(Object object) {
        if (object == null) {
            return null;
        }

        String objectId = "N/A";

        try {
            // Cố gắng lấy ID nếu đối tượng là Packet để ghi log chi tiết
            if (object instanceof Packet) {
                objectId = ((Packet) object).getPacketId();
            }

            return OBJECT_MAPPER.writeValueAsString(object);

        } catch (JsonProcessingException e) {
            logger.error("Lỗi tuần tự hóa (Serialization) đối tượng {}: {}", objectId, e.getMessage());
            return null;
        }
    }

    /**
     * Chuyển đổi chuỗi JSON String nhận được qua mạng thành đối tượng Java generic.
     * Phương thức này được sử dụng để giải mã PacketTransferWrapper.
     *
     * @param jsonString Chuỗi JSON cần giải tuần tự hóa.
     * @param clazz      Class của đối tượng đích (ví dụ:
     *                   PacketTransferWrapper.class).
     * @param <T>        Kiểu của đối tượng đích.
     * @return Đối tượng đã giải mã, hoặc null nếu xảy ra lỗi.
     */
    public static <T> T deserialize(String jsonString, Class<T> clazz) {
        if (jsonString == null || jsonString.isEmpty()) {
            logger.warn("JSON string is null or empty");
            return null;
        }
        try {
            T object = OBJECT_MAPPER.readValue(jsonString, clazz);

            // Ghi log thông báo thành công cho quá trình RPC
            if (object instanceof Packet) {
                logger.info("Successfully deserialized Packet: {}", ((Packet) object).getPacketId());
            } else {
                logger.debug("Successfully deserialized object of type: {}", clazz.getSimpleName());
            }
            return object;

        } catch (JsonProcessingException e) {
            // Log lỗi JSON để gỡ lỗi cấu trúc
            logger.error("Lỗi giải tuần tự hóa (Deserialization) JSON thành {}. Lỗi: {}",
                    clazz.getSimpleName(), e.getMessage());
            logger.debug("Chuỗi JSON thất bại: [{}]", jsonString, e);
            return null;
        }
    }

    /**
     * Phương thức tiện ích để giải mã mặc định thành Packet (duy trì tính tương
     * thích cũ).
     * Bỏ qua logic xử lý Array cũ vì nó gây nhầm lẫn.
     * 
     * @param jsonString Chuỗi JSON cần giải tuần tự hóa.
     * @return Đối tượng Packet, hoặc null nếu xảy ra lỗi.
     */
    public static Packet deserialize(String jsonString) {
        return deserialize(jsonString, Packet.class);
    }
}