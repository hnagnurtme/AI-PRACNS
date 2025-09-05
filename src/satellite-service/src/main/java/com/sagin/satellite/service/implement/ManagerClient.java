package com.sagin.satellite.service.implement;

import com.fasterxml.jackson.databind.JsonNode;
import com.fasterxml.jackson.databind.ObjectMapper;
import okhttp3.*;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;
import java.util.concurrent.TimeUnit;

public class ManagerClient {
    private static final Logger logger = LoggerFactory.getLogger(ManagerClient.class);
    private final OkHttpClient http = new OkHttpClient.Builder()
            .connectTimeout(2, TimeUnit.SECONDS)
            .readTimeout(2, TimeUnit.SECONDS)
            .build();
    private final ObjectMapper mapper = new ObjectMapper();
    private final String managerApiUrl;

    public ManagerClient(String managerApiUrl) {
        this.managerApiUrl = managerApiUrl;
    }

    public String getNodeAddress(String nodeId) {
        try {
            Request req = new Request.Builder().url(managerApiUrl + "/api/nodes").build();
            try (Response res = http.newCall(req).execute()) {
                if (!res.isSuccessful()) return null;
                JsonNode arr = mapper.readTree(res.body().string());
                for (JsonNode n : arr) {
                    if (n.path("id").asText().equals(nodeId)) {
                        String ip = n.path("ip").asText();
                        int port = n.path("port").asInt(9090);
                        return ip + ":" + port;
                    }
                }
            }
        } catch (Exception e) {
            logger.warn("ManagerClient.getNodeAddress failed", e);
        }
        return null;
    }

    public JsonNode getWeather(String region) {
        try {
            Request req = new Request.Builder().url(managerApiUrl + "/api/weather?region=" + region).build();
            try (Response res = http.newCall(req).execute()) {
                if (!res.isSuccessful()) return null;
                return mapper.readTree(res.body().string());
            }
        } catch (Exception e) {
            logger.warn("ManagerClient.getWeather failed", e);
            return null;
        }
    }
}
