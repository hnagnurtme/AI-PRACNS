package com.chatapp.auth.utils;

public class ValidateRegisterInput {
    public static boolean isValidEmail(String email) {
        return email != null && email.contains("@") && email.contains(".");
    }

    public static boolean isValidPassword(String password) {
        return password != null && password.length() >= 6;
    }

    public static String validate(String email, String password) {
        if (email == null || email.trim().isEmpty()) {
            return "Please enter email";
        }
        if (password == null || password.isEmpty()) {
            return "Please enter password";
        }
        if (!isValidEmail(email)) {
            return "Invalid email format";
        }
        if (!isValidPassword(password)) {
            return "Password must be at least 6 characters";
        }
        return null; // No errors
    }
}
