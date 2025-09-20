package com.chatapp.chatapp.helpers;

import java.net.URI;
import java.net.http.HttpRequest;
import java.net.http.HttpClient;
import java.net.http.HttpResponse;
import java.io.IOException;
import java.util.Map;

import com.chatapp.chatapp.utils.LoadProperties;
import com.google.gson.JsonObject;
import com.google.gson.JsonParser;

public class GeoLocationService {
    /** 
    * Lấy tọa độ hiện tại của client dựa trên IP
    * @return mảng [latitude, longitude]
    */
    public static double[] getCurrentLocation() throws IOException, InterruptedException {
        Map<String, String> env = LoadProperties.loadEnv();
        HttpRequest request = HttpRequest.newBuilder()
            .uri(URI.create(env.get("GEOLOCATION_API_URL")))
            .build();
        
        HttpResponse<String> response = HttpClient.newHttpClient()
            .send(request, HttpResponse.BodyHandlers.ofString());
        
        if (response.statusCode() != 200) {
            throw new RuntimeException("Failed to get location: " + response.body());
        }
        
        JsonObject json = JsonParser.parseString(response.body()).getAsJsonObject();
        double lat = json.get("lat").getAsDouble();
        double lon = json.get("lon").getAsDouble();

        return new double[] { lat, lon };
    }
}
