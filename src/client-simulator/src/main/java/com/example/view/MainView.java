package com.example.view;

import com.example.model.Packet;
import javafx.collections.ObservableList;
import javafx.geometry.Insets;
import javafx.geometry.Pos;
import javafx.scene.Node;
import javafx.scene.control.*;
import javafx.scene.layout.*;

/**
 * Modern redesigned view component with improved aesthetics and user experience
 */
public class MainView {

    // Sender controls
    public GridPane senderGrid;
    public ComboBox<String> cbSenderUsername = new ComboBox<>();
    public ComboBox<String> cbDestinationUsername = new ComboBox<>();
    public TextField tfPacketId = new TextField();
    public TextField tfSourceUserId = new TextField();
    public TextField tfDestinationUserId = new TextField();
    public TextField tfStationSource = new TextField();
    public TextField tfStationDest = new TextField();
    public TextField tfType = new TextField();
    public TextField tfAcknowledgedPacketId = new TextField();
    public TextField tfTimeSentFromSourceMs = new TextField();
    public TextArea taPayload = new TextArea();
    public TextField tfPayloadSizeByte = new TextField();
    public ComboBox<com.example.model.ServiceType> cbServiceType = new ComboBox<>();
    public Label lblQoSDetail = new Label();
    public TextField tfTTL = new TextField();
    public TextField tfCurrentHoldingNodeId = new TextField();
    public TextField tfNextHopNodeId = new TextField();
    public TextField tfPathHistory = new TextField();
    public TextField tfPriorityLevel = new TextField();
    public TextField tfPacketCount = new TextField("1");
    public TextField tfMaxAcceptableLatencyMs = new TextField();
    public TextField tfMaxAcceptableLossRate = new TextField();
    public CheckBox cbDropped = new CheckBox("Dropped");
    public TextField tfDropReason = new TextField();
    public CheckBox cbUseRL = new CheckBox("Use RL Routing");

    public Button btnSend = new Button("Send Packet");
    public TextField tfSendHost = new TextField("localhost");
    public TextField tfSendPort = new TextField("9000");

    // Receiver controls
    public TextField tfListenPort = new TextField("9000");
    public Button btnListen = new Button("Start Listening");
    public Button btnClearLog = new Button("Clear Log");
    public ListView<Packet> lvReceived = new ListView<>();
    public Label lblStatus = new Label("Ready");

    public BorderPane root = new BorderPane();

    // Color scheme
    private static final String PRIMARY_COLOR = "#2c3e50";
    private static final String ACCENT_COLOR = "#3498db";
    private static final String SUCCESS_COLOR = "#27ae60";
    private static final String BACKGROUND_COLOR = "#ecf0f1";
    private static final String CARD_COLOR = "#ffffff";

    public MainView(ObservableList<Packet> receivedItems) {
        applyModernStyles();
        buildSender();
        buildReceiver(receivedItems);

        SplitPane split = new SplitPane();
        
        // Wrap panels in styled containers
        VBox senderContainer = createStyledPanel("📤 Packet Sender", new ScrollPane(senderGrid));
        VBox receiverContainer = createStyledPanel("📥 Packet Receiver", buildReceiverPane());
        
        split.getItems().addAll(senderContainer, receiverContainer);
        split.setDividerPositions(0.35); // Sender takes 35%, Receiver takes 65% (full right side)
        split.setStyle("-fx-background-color: " + BACKGROUND_COLOR + ";");

        root.setCenter(split);
        root.setStyle("-fx-background-color: " + BACKGROUND_COLOR + ";");
    }

    private VBox createStyledPanel(String title, Node content) {
        VBox panel = new VBox(15);
        panel.setPadding(new Insets(20));
        panel.setStyle(
            "-fx-background-color: " + CARD_COLOR + ";" +
            "-fx-background-radius: 10;" +
            "-fx-effect: dropshadow(gaussian, rgba(0,0,0,0.1), 10, 0, 0, 2);"
        );
        
        Label titleLabel = new Label(title);
        titleLabel.setStyle(
            "-fx-font-size: 20px;" +
            "-fx-font-weight: bold;" +
            "-fx-text-fill: " + PRIMARY_COLOR + ";"
        );
        
        panel.getChildren().addAll(titleLabel, createDivider(), content);
        VBox.setVgrow(content, Priority.ALWAYS);
        
        VBox wrapper = new VBox(panel);
        wrapper.setPadding(new Insets(10));
        wrapper.setStyle("-fx-background-color: " + BACKGROUND_COLOR + ";");
        
        return wrapper;
    }

    private Separator createDivider() {
        Separator sep = new Separator();
        sep.setStyle("-fx-background-color: " + ACCENT_COLOR + ";");
        return sep;
    }

    private void applyModernStyles() {
        // Will be applied to individual components
    }

    private void buildSender() {
        senderGrid = new GridPane();
        senderGrid.setHgap(15);
        senderGrid.setVgap(12);
        senderGrid.setPadding(new Insets(15));

        int r = 0;
        
        // Section: User Information
        addSectionHeader(senderGrid, "👥 User Information", r++);
        
        addStyledField(senderGrid, "Sender Username", cbSenderUsername, r++, 
            "Select or type sender username");
        cbSenderUsername.setEditable(true);
        
        addStyledField(senderGrid, "Destination Username", cbDestinationUsername, r++,
            "Select or type destination username");
        cbDestinationUsername.setEditable(true);
        
        r++; // Spacing
        
        // Section: Message Content
        addSectionHeader(senderGrid, "✉️ Message Content", r++);
        
        senderGrid.add(createStyledLabel("Payload"), 0, r);
        taPayload.setPrefRowCount(4);
        taPayload.setPromptText("Enter your message here...");
        styleTextArea(taPayload);
        senderGrid.add(taPayload, 1, r++);
        
        addStyledField(senderGrid, "Packet Size (bytes)", tfPayloadSizeByte, r++,
            "Auto-calculated or enter manually");
        
        r++; // Spacing
        
        // Section: Service Configuration
        addSectionHeader(senderGrid, "⚙️ Service Configuration", r++);
        
        senderGrid.add(createStyledLabel("Service Type"), 0, r);
        cbServiceType.getItems().addAll(com.example.model.ServiceType.values());
        cbServiceType.setPrefWidth(280);
        cbServiceType.setPromptText("Select service quality type");
        styleComboBox(cbServiceType);
        senderGrid.add(cbServiceType, 1, r++);
        
        senderGrid.add(createStyledLabel("QoS Details"), 0, r);
        lblQoSDetail.setWrapText(true);
        lblQoSDetail.setMaxWidth(350);
        lblQoSDetail.setStyle(
            "-fx-font-size: 12px;" +
            "-fx-text-fill: #555;" +
            "-fx-padding: 10;" +
            "-fx-background-color: #f8f9fa;" +
            "-fx-background-radius: 5;" +
            "-fx-border-color: #dee2e6;" +
            "-fx-border-radius: 5;"
        );
        senderGrid.add(lblQoSDetail, 1, r++);
        
        r++; // Spacing
        
        // Section: Advanced Options
        addSectionHeader(senderGrid, "🎯 Sending Options", r++);
        
        addStyledField(senderGrid, "Number of Packet Pairs", tfPacketCount, r++,
            "Each pair = 1 RL + 1 non-RL (default: 1)");
        
        // Info label about automatic RL/non-RL sending
        Label infoLabel = new Label("ℹ️ Auto-sends 2 packets per pair: 1 with RL + 1 without RL (same Packet ID)");
        infoLabel.setStyle(
            "-fx-font-size: 11px;" +
            "-fx-text-fill: #856404;" +
            "-fx-padding: 8;" +
            "-fx-background-color: #fff3cd;" +
            "-fx-border-color: #ffc107;" +
            "-fx-border-radius: 5;" +
            "-fx-background-radius: 5;"
        );
        infoLabel.setWrapText(true);
        infoLabel.setMaxWidth(280);
        senderGrid.add(infoLabel, 1, r++);
        
        r++; // Spacing
        
        // Send button
        stylePrimaryButton(btnSend);
        btnSend.setPrefWidth(280);
        
        senderGrid.add(btnSend, 1, r);
    }

    private Node buildReceiverPane() {
        VBox v = new VBox(15);
        v.setPadding(new Insets(15));
        
        // Control buttons panel
        HBox controlBox = new HBox(10);
        controlBox.setAlignment(Pos.CENTER_LEFT);
        controlBox.setPadding(new Insets(10));
        controlBox.setStyle(
            "-fx-background-color: #f8f9fa;" +
            "-fx-background-radius: 5;" +
            "-fx-border-color: #dee2e6;" +
            "-fx-border-radius: 5;"
        );
        
        stylePrimaryButton(btnListen);
        btnListen.setPrefWidth(200);
        
        // Style Clear Log button
        btnClearLog.setStyle(
            "-fx-background-color: #e74c3c;" +
            "-fx-text-fill: white;" +
            "-fx-font-size: 14px;" +
            "-fx-font-weight: bold;" +
            "-fx-background-radius: 5;" +
            "-fx-padding: 10 20;" +
            "-fx-cursor: hand;"
        );
        btnClearLog.setPrefWidth(150);
        
        btnClearLog.setOnMouseEntered(e -> btnClearLog.setStyle(
            "-fx-background-color: #c0392b;" +
            "-fx-text-fill: white;" +
            "-fx-font-size: 14px;" +
            "-fx-font-weight: bold;" +
            "-fx-background-radius: 5;" +
            "-fx-padding: 10 20;" +
            "-fx-cursor: hand;" +
            "-fx-effect: dropshadow(gaussian, rgba(0,0,0,0.3), 5, 0, 0, 2);"
        ));
        
        btnClearLog.setOnMouseExited(e -> btnClearLog.setStyle(
            "-fx-background-color: #e74c3c;" +
            "-fx-text-fill: white;" +
            "-fx-font-size: 14px;" +
            "-fx-font-weight: bold;" +
            "-fx-background-radius: 5;" +
            "-fx-padding: 10 20;" +
            "-fx-cursor: hand;"
        ));
        
        controlBox.getChildren().addAll(btnListen, btnClearLog);
        
        // Received packets list
        Label listLabel = createStyledLabel("📨 Received Packets");
        listLabel.setStyle("-fx-font-size: 14px; -fx-font-weight: bold;");
        
        lvReceived.setStyle(
            "-fx-background-color: white;" +
            "-fx-border-color: #dee2e6;" +
            "-fx-border-radius: 5;" +
            "-fx-background-radius: 5;"
        );
        
        // Status label
        lblStatus.setStyle(
            "-fx-font-size: 12px;" +
            "-fx-padding: 8;" +
            "-fx-background-color: #d4edda;" +
            "-fx-text-fill: #155724;" +
            "-fx-background-radius: 5;"
        );
        
        v.getChildren().addAll(controlBox, listLabel, lvReceived, lblStatus);
        VBox.setVgrow(lvReceived, Priority.ALWAYS);
        
        return v;
    }

    private void buildReceiver(ObservableList<Packet> receivedItems) {
        lvReceived.setItems(receivedItems);
        lvReceived.setCellFactory(list -> new ListCell<>() {
            @Override
            protected void updateItem(Packet item, boolean empty) {
                super.updateItem(item, empty);
                if (empty || item == null) {
                    setText(null);
                    setGraphic(null);
                    setStyle("");
                } else {
                    // Determine if packet uses RL or not
                    boolean usesRL = item.isUseRL(); // Assuming Packet has isUseRL() method
                    
                    VBox cell = new VBox(5);
                    cell.setPadding(new Insets(10));
                    
                    // Different styling for RL vs non-RL packets
                    if (usesRL) {
                        cell.setStyle(
                            "-fx-background-color: linear-gradient(to right, #e8f5e9, #c8e6c9);" +
                            "-fx-background-radius: 8;" +
                            "-fx-border-color: #4caf50;" +
                            "-fx-border-width: 2;" +
                            "-fx-border-radius: 8;" +
                            "-fx-effect: dropshadow(gaussian, rgba(76, 175, 80, 0.3), 5, 0, 0, 1);"
                        );
                    } else {
                        cell.setStyle(
                            "-fx-background-color: linear-gradient(to right, #fff3e0, #ffe0b2);" +
                            "-fx-background-radius: 8;" +
                            "-fx-border-color: #ff9800;" +
                            "-fx-border-width: 2;" +
                            "-fx-border-radius: 8;" +
                            "-fx-effect: dropshadow(gaussian, rgba(255, 152, 0, 0.3), 5, 0, 0, 1);"
                        );
                    }
                    
                    // RL Badge
                    Label badge = new Label(usesRL ? "🤖 RL ROUTING" : "📍 STATIC ROUTING");
                    badge.setStyle(
                        "-fx-font-size: 10px;" +
                        "-fx-font-weight: bold;" +
                        "-fx-padding: 3 8;" +
                        "-fx-background-radius: 3;" +
                        (usesRL ? 
                            "-fx-background-color: #4caf50; -fx-text-fill: white;" : 
                            "-fx-background-color: #ff9800; -fx-text-fill: white;")
                    );
                    
                    Label header = new Label(String.format("📦 Packet ID: %s", item.getPacketId()));
                    header.setStyle(
                        "-fx-font-weight: bold;" +
                        "-fx-font-size: 13px;" +
                        (usesRL ? "-fx-text-fill: #2e7d32;" : "-fx-text-fill: #e65100;")
                    );
                    
                    Label route = new Label(String.format("🔀 %s → %s", 
                        item.getSourceUserId(), item.getDestinationUserId()));
                    route.setStyle("-fx-font-size: 11px; -fx-text-fill: #555;");
                    
                    Label payload = new Label(String.format("💬 %s", item.getDecodedPayload()));
                    payload.setStyle("-fx-font-size: 12px; -fx-text-fill: #333;");
                    payload.setWrapText(true);
                    
                    HBox headerRow = new HBox(10, badge, header);
                    headerRow.setAlignment(Pos.CENTER_LEFT);
                    
                    cell.getChildren().addAll(headerRow, route, payload);
                    
                    setText(null);
                    setGraphic(cell);
                    setStyle("-fx-padding: 5; -fx-background-color: transparent;");
                }
            }
        });
    }

    // Styling helper methods
    private void addSectionHeader(GridPane grid, String text, int row) {
        Label header = new Label(text);
        header.setStyle(
            "-fx-font-size: 15px;" +
            "-fx-font-weight: bold;" +
            "-fx-text-fill: " + PRIMARY_COLOR + ";" +
            "-fx-padding: 10 0 5 0;"
        );
        grid.add(header, 0, row, 2, 1);
    }

    private Label createStyledLabel(String text) {
        Label label = new Label(text);
        label.setStyle(
            "-fx-font-size: 13px;" +
            "-fx-text-fill: " + PRIMARY_COLOR + ";" +
            "-fx-font-weight: 500;"
        );
        return label;
    }

    private void addStyledField(GridPane grid, String labelText, TextField field, int row, String prompt) {
        grid.add(createStyledLabel(labelText), 0, row);
        field.setPromptText(prompt);
        styleTextField(field);
        grid.add(field, 1, row);
    }

    private void addStyledField(GridPane grid, String labelText, ComboBox<?> combo, int row, String prompt) {
        grid.add(createStyledLabel(labelText), 0, row);
        combo.setPromptText(prompt);
        styleComboBox(combo);
        grid.add(combo, 1, row);
    }

    private void styleTextField(TextField field) {
        field.setStyle(
            "-fx-background-color: white;" +
            "-fx-border-color: #ced4da;" +
            "-fx-border-radius: 5;" +
            "-fx-background-radius: 5;" +
            "-fx-padding: 8;" +
            "-fx-font-size: 13px;"
        );
        field.setPrefWidth(280);
        
        field.focusedProperty().addListener((obs, oldVal, newVal) -> {
            if (newVal) {
                field.setStyle(
                    "-fx-background-color: white;" +
                    "-fx-border-color: " + ACCENT_COLOR + ";" +
                    "-fx-border-width: 2;" +
                    "-fx-border-radius: 5;" +
                    "-fx-background-radius: 5;" +
                    "-fx-padding: 8;" +
                    "-fx-font-size: 13px;"
                );
            } else {
                field.setStyle(
                    "-fx-background-color: white;" +
                    "-fx-border-color: #ced4da;" +
                    "-fx-border-radius: 5;" +
                    "-fx-background-radius: 5;" +
                    "-fx-padding: 8;" +
                    "-fx-font-size: 13px;"
                );
            }
        });
    }

    private void styleTextArea(TextArea area) {
        area.setStyle(
            "-fx-background-color: white;" +
            "-fx-border-color: #ced4da;" +
            "-fx-border-radius: 5;" +
            "-fx-background-radius: 5;" +
            "-fx-padding: 8;" +
            "-fx-font-size: 13px;"
        );
        area.setPrefWidth(280);
        
        area.focusedProperty().addListener((obs, oldVal, newVal) -> {
            if (newVal) {
                area.setStyle(
                    "-fx-background-color: white;" +
                    "-fx-border-color: " + ACCENT_COLOR + ";" +
                    "-fx-border-width: 2;" +
                    "-fx-border-radius: 5;" +
                    "-fx-background-radius: 5;" +
                    "-fx-padding: 8;" +
                    "-fx-font-size: 13px;"
                );
            } else {
                area.setStyle(
                    "-fx-background-color: white;" +
                    "-fx-border-color: #ced4da;" +
                    "-fx-border-radius: 5;" +
                    "-fx-background-radius: 5;" +
                    "-fx-padding: 8;" +
                    "-fx-font-size: 13px;"
                );
            }
        });
    }

    private void styleComboBox(ComboBox<?> combo) {
        combo.setStyle(
            "-fx-background-color: white;" +
            "-fx-border-color: #ced4da;" +
            "-fx-border-radius: 5;" +
            "-fx-background-radius: 5;" +
            "-fx-padding: 5;" +
            "-fx-font-size: 13px;"
        );
        combo.setPrefWidth(280);
    }

    private void stylePrimaryButton(Button btn) {
        btn.setStyle(
            "-fx-background-color: " + ACCENT_COLOR + ";" +
            "-fx-text-fill: white;" +
            "-fx-font-size: 14px;" +
            "-fx-font-weight: bold;" +
            "-fx-background-radius: 5;" +
            "-fx-padding: 10 20;" +
            "-fx-cursor: hand;"
        );
        
        btn.setOnMouseEntered(e -> btn.setStyle(
            "-fx-background-color: #2980b9;" +
            "-fx-text-fill: white;" +
            "-fx-font-size: 14px;" +
            "-fx-font-weight: bold;" +
            "-fx-background-radius: 5;" +
            "-fx-padding: 10 20;" +
            "-fx-cursor: hand;" +
            "-fx-effect: dropshadow(gaussian, rgba(0,0,0,0.3), 5, 0, 0, 2);"
        ));
        
        btn.setOnMouseExited(e -> btn.setStyle(
            "-fx-background-color: " + ACCENT_COLOR + ";" +
            "-fx-text-fill: white;" +
            "-fx-font-size: 14px;" +
            "-fx-font-weight: bold;" +
            "-fx-background-radius: 5;" +
            "-fx-padding: 10 20;" +
            "-fx-cursor: hand;"
        ));
    }
}