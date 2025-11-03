package com.example;

import com.example.controller.MainController;
import com.example.view.MainView;
import javafx.application.Application;
import javafx.scene.Scene;
import javafx.stage.Stage;

/**
 * Main JavaFX application launching the MVC components.
 * - MainView constructs the UI
 * - MainController wires behaviors and model interactions
 */
public class MainApp extends Application {

    @Override
    public void start(Stage primaryStage) {
        // ObservableList for received packets is created inside the view/controller combo
    javafx.collections.ObservableList<com.example.model.Packet> list = javafx.collections.FXCollections.observableArrayList();
    MainView view = new MainView(list);
    new MainController(view);

        Scene scene = new Scene(view.root, 1100, 720);
        primaryStage.setTitle("Packet Sender / Receiver - MVC Example");
        primaryStage.setScene(scene);
        primaryStage.show();
    }

    public static void main(String[] args) {
        launch(args);
    }
}
