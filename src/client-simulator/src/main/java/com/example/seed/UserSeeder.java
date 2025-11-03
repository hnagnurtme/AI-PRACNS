package com.example.seed;

import com.example.factory.PositionFactory;
import com.example.model.UserInfo;
import com.example.repository.MongoUserRepository;

import java.util.ArrayList;
import java.util.List;
import java.util.Map;

public class UserSeeder {

    public static void main(String[] args) {
        try (MongoUserRepository userRepo = new MongoUserRepository()) {

            // --- 2. Lấy danh sách 50 thành phố lớn ---
            Map<String, ?> cities = PositionFactory.createWorldCities();
            int port = 10000;

            List<UserInfo> users = new ArrayList<>();
            for (String cityName : cities.keySet()) {
                UserInfo user = new UserInfo();
                // ID theo user-city
                user.setUserId("user-" + cityName.replaceAll("\\s+", ""));
                user.setUserName("User_" + cityName.replaceAll("\\s+", ""));

                // IP và port ví dụ
                user.setIpAddress("127.0.0.1");
                
                user.setPort(port++); // tất cả cùng port ví dụ, bạn có thể random nếu muốn

                user.setCityName(cityName);

                users.add(user);
            }

            // --- 3. Bulk insert / update vào MongoDB ---
            userRepo.bulkUpdateUsers(users);

            System.out.println("Seed dữ liệu thành công: " + users.size() + " users");

        } catch (Exception e) {
            e.printStackTrace();
        }
    }
}
