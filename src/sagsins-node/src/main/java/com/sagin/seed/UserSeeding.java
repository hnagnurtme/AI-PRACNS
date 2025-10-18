package com.sagin.seed;

import com.sagin.model.*;
import com.sagin.repository.MongoUserRepository;

import java.util.ArrayList;
import java.util.List;

/**
 * Seed 5 users with TCP + 127.0.0.1
 */
public class UserSeeding {

    public static void main(String[] args) {
        List<UserInfo> users = new ArrayList<>();

        users.add(createUser("user-01", "Alice", 10.0, 100.0));
        users.add(createUser("user-02", "Bob", 20.0, 110.0));
        users.add(createUser("user-03", "Charlie", 15.0, 105.0));
        users.add(createUser("user-04", "David", 25.0, 120.0));
        users.add(createUser("user-05", "Eve", 5.0, 95.0));

        try (MongoUserRepository repo = new MongoUserRepository()) {
            repo.bulkUpdateUsers(users);
            System.out.println("Seeded 5 users successfully.");
        } catch (Exception e) {
            e.printStackTrace();
        }
    }

    private static UserInfo createUser(String userId, String userName, double lat, double lon) {
        UserInfo user = new UserInfo();
        user.setUserId(userId);
        user.setUserName(userName);

        // Position
        user.setPosition(new Position(lat, lon, 0));

        // Communication TCP
        user.setCommunication(new Communication(
                2.2,   // bandwidth
                100,   // maxThroughput
                20,    // latency
                30,    // jitter
                60,    // timeout
                2000,  // bufferSize
                10,    // retry
                "127.0.0.1",
                7000 + Integer.parseInt(userId.split("-")[1]), // assign unique port
                "TCP"
        ));

        return user;
    }
}
