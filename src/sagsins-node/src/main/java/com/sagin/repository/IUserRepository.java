package com.sagin.repository;

import com.sagin.model.UserInfo;

import java.util.List;
import java.util.Optional;

/**
 * Interface Repository để truy cập dữ liệu người dùng (End-Users).
 */
public interface IUserRepository {

    /**
     * Tìm thông tin người dùng (bao gồm IP/Port) bằng userId của họ.
     * @param userId ID của người dùng (ví dụ: "USER-02")
     * @return Optional chứa UserInfo nếu tìm thấy
     */
    Optional<UserInfo> findByUserId(String userId);


    void bulkUpdateUsers(List<UserInfo> users);
    /**
     * Đóng kết nối đến CSDL.
     */
    void close();
}