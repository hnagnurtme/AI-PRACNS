package com.sagin.seed;

import com.sagin.model.*;
import com.sagin.repository.MongoUserRepository;

import java.util.ArrayList;
import java.util.List;

/**
 * Seed 10 globally distributed users with TCP + 127.0.0.1
 * Each user assigned unique port (7001–7010)
 */
public class UserSeedingGlobal {

    public static void main(String[] args) {
        List<UserInfo> users = new ArrayList<>();

        users.add(createUser("user-tokyo", "User Tokyo", 35.6895, 139.6917, 7001));      // Asia
        users.add(createUser("user-paris", "User Paris", 48.8566, 2.3522, 7002));        // Europe
        users.add(createUser("user-newyork", "User New York", 40.7128, -74.0060, 7003)); // North America
        users.add(createUser("user-sydney", "User Sydney", -33.8688, 151.2093, 7004));   // Oceania
        users.add(createUser("user-saopaulo", "User Sao Paulo", -23.5505, -46.6333, 7005)); // South America
        users.add(createUser("user-moscow", "User Moscow", 55.7558, 37.6173, 7006));     // Europe/Asia
        users.add(createUser("user-nairobi", "User Nairobi", -1.2921, 36.8219, 7007));   // Africa
        users.add(createUser("user-mexico", "User Mexico City", 19.4326, -99.1332, 7008)); // Central America
        users.add(createUser("user-delhi", "User New Delhi", 28.6139, 77.2090, 7009));   // Asia
        users.add(createUser("user-london", "User London", 51.5074, -0.1278, 7010));     // Europe

        try (MongoUserRepository repo = new MongoUserRepository()) {
            repo.bulkUpdateUsers(users);
            System.out.println("✅ Seeded 10 globally distributed users successfully into MongoDB.");
        } catch (Exception e) {
            e.printStackTrace();
        }
    }

    private static UserInfo createUser(String userId, String userName, double lat, double lon, int port) {
        UserInfo user = new UserInfo();
        user.setUserId(userId);
        user.setUserName(userName);
        user.setPosition(new Position(lat, lon, 0));

        user.setCommunication(new Communication(
                2.2,   // bandwidth (MHz)
                100,   // maxThroughput (Mbps)
                20,    // latency (ms)
                30,    // jitter (ms)
                60,    // timeout (s)
                2000,  // bufferSize (KB)
                10,    // retry
                "127.0.0.1",
                port,
                "TCP"
        ));

        return user;
    }
}
