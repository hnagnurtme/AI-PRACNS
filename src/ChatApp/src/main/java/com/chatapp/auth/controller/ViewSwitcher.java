package com.chatapp.auth.controller;

import java.net.URL;    

import javafx.fxml.FXMLLoader;
import javafx.scene.Parent;
import javafx.scene.Scene;
import javafx.stage.Stage;
import javafx.stage.StageStyle;

public class ViewSwitcher {
    private static Stage currentStage;

    public static void setStage(Stage stage) {
        currentStage = stage;
    }
    
    public static void switchTo(String fxmlFile) throws Exception {
        if (currentStage == null) {
            currentStage = new Stage();
            currentStage.initStyle(StageStyle.UTILITY);
        }

        URL url = ViewSwitcher.class.getResource("/com/chatapp/auth/" + fxmlFile);
        if (url == null) {
            throw new RuntimeException("FXML file not found: " + fxmlFile);
        }

        Parent root = FXMLLoader.load(url);
        
        Scene scene = currentStage.getScene();
        scene.setRoot(root);
        currentStage.setScene(scene);
        currentStage.centerOnScreen();
    }
}
