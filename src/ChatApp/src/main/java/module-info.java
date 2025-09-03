module com.chatapp.chatapp {
    requires javafx.controls;
    requires javafx.fxml;

    requires org.controlsfx.controls;
    requires com.dlsc.formsfx;
    requires org.kordamp.bootstrapfx.core;

    opens com.chatapp.chatapp to javafx.fxml;
    exports com.chatapp.chatapp;
    exports com.chatapp.chatapp.controller;
    opens com.chatapp.chatapp.controller to javafx.fxml;
}