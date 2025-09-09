package com.chatapp.auth.model.services;

import java.io.IOException;
import java.nio.charset.StandardCharsets;
import java.util.HashMap;
import java.util.Map;

import org.apache.hc.client5.http.classic.methods.HttpPost;
import org.apache.hc.client5.http.impl.classic.CloseableHttpClient;
import org.apache.hc.client5.http.impl.classic.CloseableHttpResponse;
import org.apache.hc.client5.http.impl.classic.HttpClients;
import org.apache.hc.core5.http.io.entity.StringEntity;

import com.chatapp.auth.utils.PropertiesUtil;
import com.chatapp.chatapp.config.FirebaseConfig;
import com.fasterxml.jackson.databind.JsonNode;
import com.fasterxml.jackson.databind.ObjectMapper;
import com.google.cloud.firestore.DocumentReference;
import com.google.cloud.firestore.FieldValue;
import com.google.cloud.firestore.Firestore;

public class FirebaseAuthService {
    private static final String API_KEY = PropertiesUtil.getString("firebase.api.key");
    private static final String SIGNUP_URL =
            "https://identitytoolkit.googleapis.com/v1/accounts:signUp?key=" + API_KEY;
    private static final String SIGNIN_URL =
            "https://identitytoolkit.googleapis.com/v1/accounts:signInWithPassword?key=" + API_KEY; 
    
    private final ObjectMapper mapper = new ObjectMapper();

    public String register(String email, String password) throws Exception {
        return sendAuthRequest(SIGNUP_URL, email, password, true);
    }

    public String login(String email, String password) throws Exception {
        return sendAuthRequest(SIGNIN_URL, email, password, false);
    }

    private String sendAuthRequest(String url, String email, String password, boolean isRegister) throws Exception {
        try (CloseableHttpClient client = HttpClients.createDefault()) {
            HttpPost request = new HttpPost(url);
            request.setHeader("Content-Type", "application/json");

            // Create request body
            Map<String, Object> requestBody = new HashMap<>();
            requestBody.put("email", email.trim());
            requestBody.put("password", password);
            requestBody.put("returnSecureToken", true);

            // Convert to JSON
            String jsonBody = mapper.writeValueAsString(requestBody);
            request.setEntity(new StringEntity(jsonBody, StandardCharsets.UTF_8));

            // Execute request
            try (CloseableHttpResponse response = client.execute(request)) {
                String responseBody = response.getEntity() != null ? 
                    new String(response.getEntity().getContent().readAllBytes(), StandardCharsets.UTF_8) : "";
                
                if (response.getCode() >= 200 && response.getCode() < 300) {
                    // Parse successful response
                    JsonNode jsonNode = mapper.readTree(responseBody);
                    if (jsonNode.has("idToken")) {
                        String idToken = jsonNode.get("idToken").asText();
                        if (isRegister) {
                            saveUserToFirestore(idToken, email);
                        }
                        return idToken;
                    } else {
                        throw new Exception("Cannot get token from server");
                    }
                } else {
                    // Parse error response
                    throw new Exception(parseFirebaseError(responseBody));
                }
            }
        } catch (Exception e) {
            if (e instanceof Exception) {
                throw e;
            }
            throw new Exception("Network error: " + e.getMessage(), e);
        }
    }

    private String parseFirebaseError(String responseBody) throws Exception {
        try {
            JsonNode errorNode = mapper.readTree(responseBody);
            if (errorNode.has("error") && errorNode.get("error").has("message")) {
                String errorCode = errorNode.get("error").get("message").asText();
                
                // Basic error translation
                return switch (errorCode) {
                    case "EMAIL_EXISTS" -> "Email already exists";
                    case "INVALID_LOGIN_CREDENTIALS" -> "Email or password is incorrect";
                    case "EMAIL_NOT_FOUND" -> "Email not found";
                    case "INVALID_PASSWORD" -> "Password is incorrect";
                    case "WEAK_PASSWORD" -> "Password is too weak (minimum 6 characters)";
                    case "INVALID_EMAIL" -> "Invalid email format";
                    case "TOO_MANY_ATTEMPTS_TRY_LATER" -> "Too many attempts. Try again later";
                    case "USER_DISABLED" -> "Account is locked";
                    default -> "Authentication failed: " + errorCode;
                };
            }
        } catch (Exception e) {
            // Ignore JSON parsing errors
            e.printStackTrace();
        }
        return "An error occurred";
    }

    private void saveUserToFirestore(String uid, String email) throws Exception {
        Firestore db = FirebaseConfig.getFirestore();
        DocumentReference docRef = db.collection("users").document(uid);

        Map<String, Object> userData = new HashMap<>();
        userData.put("email", email);
        userData.put("createdAt", FieldValue.serverTimestamp());

        docRef.set(userData)
            .get();

        System.out.println("User saved to Firestore: " + email);
    }
}
