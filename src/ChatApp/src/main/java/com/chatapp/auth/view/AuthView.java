package com.chatapp.auth.view;

import javafx.application.Application;
import javafx.fxml.FXMLLoader;
import javafx.scene.Parent;
import javafx.scene.Scene;
import javafx.stage.Stage;

public class AuthView extends Application {
    private static final String APP_TITLE = "SAGIN Network Client - Login";
    private static final int MIN_WIDTH = 400;
    private static final int MIN_HEIGHT = 500;

    @Override
    public void start(Stage stage) throws Exception {
        FXMLLoader loader = new FXMLLoader(getClass().getResource("/com/chatapp/auth/LoginView.fxml"));
        Parent root = loader.load();

        Scene scene = new Scene(root, MIN_WIDTH, MIN_HEIGHT);

        stage.setTitle(APP_TITLE);
        stage.setScene(scene);
        stage.setResizable(false);
        stage.centerOnScreen();

        stage.show();
    }

    public static void main(String[] args) {
        launch(args);
    }
}
