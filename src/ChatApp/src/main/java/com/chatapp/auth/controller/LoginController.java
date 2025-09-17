package com.chatapp.auth.controller;

import com.chatapp.auth.model.entities.User;
import com.chatapp.auth.model.entities.UserSession;
import com.chatapp.auth.model.services.FirebaseAuthService;
import com.chatapp.chatapp.view.JavaFxApp;

import javafx.application.Platform;
import javafx.concurrent.Task;
import javafx.fxml.FXML;
import javafx.fxml.Initializable;
import javafx.scene.control.*;
import javafx.stage.Stage;

import java.net.URL;
import java.util.ResourceBundle;

public class LoginController implements Initializable {
    @FXML private TextField emailField;
    @FXML private PasswordField passwordField;
    @FXML private Label messageLabel;
    @FXML private Button loginButton;
    @FXML private Button registerButton;

    private final FirebaseAuthService authService = new FirebaseAuthService();

    @Override
    public void initialize(URL url, ResourceBundle resourceBundle) {
        Platform.runLater(() -> {
            if (loginButton.getScene() != null) {
                Stage stage = (Stage) loginButton.getScene().getWindow();
                ViewSwitcher.setStage(stage);
            }
        });
    }

    @FXML
    private void handleLogin() {
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

        // Disable UI during login
        setUIEnabled(false);
        showInfo("Logging in...");

        // Login in background thread
        Task<String> loginTask = new Task<String>() {
            @Override
            protected String call() throws Exception {
                return authService.login(email, password);
            }

            @Override
            protected void succeeded() {
                Platform.runLater(() -> {
                    String token = getValue();
                    User currentUser = new User(email, token, 0, 0);
                    showSuccess("Login successful!");
                    openMainApp(currentUser);
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

        Thread loginThread = new Thread(loginTask);
        loginThread.setDaemon(true);
        loginThread.start();
    }

    @FXML
    private void goToRegister() {
        try {
            ViewSwitcher.switchTo("RegisterView.fxml");
        } catch (Exception e) {
            showError("Cannot switch to register page");
        }
    }

    private void openMainApp(User user) {
        try {
            UserSession.setCurrentUser(user);
            
            Stage mainStage = new Stage();
            JavaFxApp mainApp = new JavaFxApp();
            mainApp.start(mainStage);
            
            Stage currentStage = (Stage) loginButton.getScene().getWindow();
            currentStage.close();
            
        } catch (Exception e) {
            showError("Cannot open main application");
            setUIEnabled(true);
        }
    }

    private void setUIEnabled(boolean enabled) {
        loginButton.setDisable(!enabled);
        registerButton.setDisable(!enabled);
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
