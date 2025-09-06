package com.chatapp.chatapp.view;

import javafx.application.Application;
import javafx.fxml.FXMLLoader;
import javafx.scene.Parent;
import javafx.scene.Scene;
import javafx.stage.Stage;
public class JavaFxApp extends Application {
    private static final String APP_TITLE = "SAGIN Network Client";
    private static final int MIN_WIDTH = 1400;
    private static final int MIN_HEIGHT = 900;

    @Override
    public void start(Stage stage) throws Exception {
        // Load main window FXML
        FXMLLoader loader = new FXMLLoader(getClass().getResource("/fxml/main-window.fxml"));
        Parent root = loader.load();

        // Create scene
        Scene scene = new Scene(root, MIN_WIDTH, MIN_HEIGHT);
        
        // Load CSS
        scene.getStylesheets().add(getClass().getResource("/css/main-style.css").toExternalForm());

        // Configure stage
        stage.setTitle(APP_TITLE);
        stage.setScene(scene);
        stage.setMinWidth(MIN_WIDTH);
        stage.setMinHeight(MIN_HEIGHT);
        stage.setMaximized(true);

        // Show stage
        stage.show();
    }

    public static void main(String[] args) {
        launch(args);
    }
} 

