package com.sagin.satellite;

import com.google.cloud.firestore.Firestore;
import com.sagin.satellite.config.FireStoreConfiguration;
import com.sagin.satellite.model.Packet;;

public class SatelliteApp {
    public static void main(String[] args) throws Exception {
        FireStoreConfiguration.init();

        Packet packet = new Packet(
                "p1",
                "SAT-001",
                "GROUND-001",
                512,
                10,
                "SAT-001",
                "SAT-002",
                System.currentTimeMillis(),
                System.currentTimeMillis(),
                25.3,
                0.01,
                false);

        Firestore db = FireStoreConfiguration.getFirestore();
        db.collection("packets")
                .document(packet.getId())
                .set(packet)
                .get();

        // Tắt gọn gàng
        FireStoreConfiguration.shutdown();
    }
}