module com.chatapp.chatapp {
    requires javafx.controls;
    requires javafx.fxml;

    requires org.controlsfx.controls;
    requires com.dlsc.formsfx;
    requires org.kordamp.bootstrapfx.core;

    exports com.chatapp.chatapp.controller;
    opens com.chatapp.chatapp.controller to javafx.fxml;

    exports com.chatapp.chatapp.view;
    opens com.chatapp.chatapp.view to javafx.fxml;

    exports com.chatapp.chatapp.model.interfaces;
    exports com.chatapp.chatapp.model.services;
}