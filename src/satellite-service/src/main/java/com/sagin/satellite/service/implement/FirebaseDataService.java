package com.sagin.satellite.service.implement;

import com.google.firebase.database.DatabaseReference;
import com.google.firebase.database.FirebaseDatabase;

public class FirebaseDataService {

    public static void addTestData() {
        // Lấy reference tới root của Realtime Database
        DatabaseReference ref = FirebaseDatabase.getInstance().getReference("test-data");

        // Tạo object mẫu
        User user = new User("u001", "Trung Ánh", "dev@sagin.com");

        // Push vào database (sẽ tự sinh key unique)
        ref.push().setValueAsync(user);

        System.out.println("✅ Data pushed to Firebase Realtime Database!");
    }

    // Một class nhỏ làm model
    static class User {
        public String id;
        public String name;
        public String email;

        // Firebase yêu cầu constructor rỗng
        public User() {}

        public User(String id, String name, String email) {
            this.id = id;
            this.name = name;
            this.email = email;
        }
    }
}