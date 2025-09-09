package com.chatapp.auth.controller;

import javafx.application.Platform;
import javafx.concurrent.Task;
import javafx.fxml.FXML;
import javafx.scene.control.*;

import com.chatapp.auth.model.services.FirebaseAuthService;

public class RegisterController {
    @FXML private TextField emailField;
    @FXML private PasswordField passwordField;
    @FXML private Label messageLabel;
    @FXML private Button registerButton;
    @FXML private Button loginButton;
    
    private final FirebaseAuthService authService = new FirebaseAuthService();

    @FXML
    private void handleRegister() {
        String email = emailField.getText().trim();
        String password = passwordField.getText();

        // Basic validation
        if (email.isEmpty()) {
            showError("Please enter email");
            return;
        }
        if (password.isEmpty()) {
            showError("Please enter password");
            return;
        }
        if (!email.contains("@")) {
            showError("Invalid email format");
            return;
        }
        if (password.length() < 6) {
            showError("Password must be at least 6 characters");
            return;
        }

        // Disable UI during registration
        setUIEnabled(false);
        showInfo("Creating account...");

        // Register in background thread
        Task<String> registerTask = new Task<String>() {
            @Override
            protected String call() throws Exception {
                return authService.register(email, password);
            }

            @Override
            protected void succeeded() {
                Platform.runLater(() -> {
                    showSuccess("Registration successful! Please login.");
                    // Auto switch to login after 2 seconds
                    new Thread(() -> {
                        try {
                            Thread.sleep(2000);
                            Platform.runLater(() -> goToLogin());
                        } catch (InterruptedException e) {
                            // Ignore
                        }
                    }).start();
                });
            }

            @Override
            protected void failed() {
                Platform.runLater(() -> {
                    String errorMessage = getException().getMessage();
                    showError(errorMessage);
                    setUIEnabled(true);
                });
            }
        };

        Thread registerThread = new Thread(registerTask);
        registerThread.setDaemon(true);
        registerThread.start();
    }

    @FXML
    private void goToLogin() {
        try {
            ViewSwitcher.switchTo("LoginView.fxml");
        } catch (Exception e) {
            showError("Cannot switch to login page");
        }
    }

    private void setUIEnabled(boolean enabled) {
        registerButton.setDisable(!enabled);
        loginButton.setDisable(!enabled);
        emailField.setDisable(!enabled);
        passwordField.setDisable(!enabled);
    }

    private void showError(String message) {
        messageLabel.setText(message);
        messageLabel.setStyle("-fx-text-fill: red;");
    }

    private void showSuccess(String message) {
        messageLabel.setText(message);
        messageLabel.setStyle("-fx-text-fill: green;");
    }

    private void showInfo(String message) {
        messageLabel.setText(message);
        messageLabel.setStyle("-fx-text-fill: blue;");
    }
}

