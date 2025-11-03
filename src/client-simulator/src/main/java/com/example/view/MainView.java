package com.example.view;

import com.example.model.Packet;
import javafx.collections.ObservableList;
import javafx.geometry.Insets;
import javafx.scene.Node;
import javafx.scene.control.*;
import javafx.scene.layout.BorderPane;
import javafx.scene.layout.GridPane;
import javafx.scene.layout.HBox;
import javafx.scene.layout.Priority;
import javafx.scene.layout.VBox;

/**
 * View component that builds JavaFX nodes for the Sender Panel and Receiver Panel.
 * This class does not handle behavior; it only exposes controls so a Controller
 * can wire up actions.
 */
public class MainView {

    // Sender controls
    public GridPane senderGrid;
    public javafx.scene.control.ComboBox<String> cbSenderUsername = new javafx.scene.control.ComboBox<>();
    public javafx.scene.control.ComboBox<String> cbDestinationUsername = new javafx.scene.control.ComboBox<>();
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
    public javafx.scene.control.ComboBox<com.example.model.ServiceType> cbServiceType = new javafx.scene.control.ComboBox<>();
    public Label lblQoSDetail = new Label();
    public TextField tfTTL = new TextField();
    public TextField tfCurrentHoldingNodeId = new TextField();
    public TextField tfNextHopNodeId = new TextField();
    public TextField tfPathHistory = new TextField();
    public TextField tfPriorityLevel = new TextField();
    public CheckBox cbUseRL = new CheckBox("Use RL");
    public TextField tfMaxAcceptableLatencyMs = new TextField();
    public TextField tfMaxAcceptableLossRate = new TextField();
    public CheckBox cbDropped = new CheckBox("Dropped");
    public TextField tfDropReason = new TextField();

    public Button btnSend = new Button("Send");
    public TextField tfSendHost = new TextField("localhost");
    public TextField tfSendPort = new TextField("9000");

    // Receiver controls
    public TextField tfListenPort = new TextField("9000");
    public Button btnListen = new Button("Listen");
    public ListView<Packet> lvReceived = new ListView<>();
    public Label lblStatus = new Label("Ready");

    public BorderPane root = new BorderPane();

    public MainView(ObservableList<Packet> receivedItems) {
        buildSender();
        buildReceiver(receivedItems);

        SplitPane split = new SplitPane();
        split.getItems().addAll(new ScrollPane(senderGrid), buildReceiverPane());
        split.setDividerPositions(0.55);

        root.setCenter(split);
    }

    private void buildSender() {
        senderGrid = new GridPane();
        senderGrid.setHgap(8);
        senderGrid.setVgap(6);
        senderGrid.setPadding(new Insets(10));

    int r = 0;
    // Username selection for sender and destination (auto-fills user IDs, host/port, stations)
    senderGrid.add(new Label("Sender Username"), 0, r); 
    cbSenderUsername.setEditable(true); 
    cbSenderUsername.setPromptText("Select or type sender username");
    senderGrid.add(cbSenderUsername, 1, r++);
    
    senderGrid.add(new Label("Destination Username"), 0, r); 
    cbDestinationUsername.setEditable(true);
    cbDestinationUsername.setPromptText("Select or type destination username");
    senderGrid.add(cbDestinationUsername, 1, r++);
    
    // Separator
    senderGrid.add(new Separator(), 0, r, 2, 1); r++;
    
    // Payload and message details
    senderGrid.add(new Label("Payload (plain text)"), 0, r); 
    taPayload.setPrefRowCount(4); 
    taPayload.setPromptText("Enter message content");
    senderGrid.add(taPayload, 1, r++);
    
    senderGrid.add(new Label("Packet Size (bytes)"), 0, r);
    tfPayloadSizeByte.setPromptText("Enter packet size in bytes");
    senderGrid.add(tfPayloadSizeByte, 1, r++);
    
    senderGrid.add(new Label("Service Type"), 0, r);
    cbServiceType.getItems().addAll(com.example.model.ServiceType.values());
    cbServiceType.setPrefWidth(240);
    cbServiceType.setPromptText("Select service type");
    senderGrid.add(cbServiceType, 1, r++);
    
    senderGrid.add(new Label("QoS Details"), 0, r); 
    lblQoSDetail.setWrapText(true);
    lblQoSDetail.setMaxWidth(300);
    lblQoSDetail.setStyle("-fx-font-size: 11px; -fx-text-fill: #333; -fx-padding: 5;");
    senderGrid.add(lblQoSDetail, 1, r++);
    
    // Optional fields
    senderGrid.add(new Label("TTL (optional)"), 0, r); 
    tfTTL.setPromptText("Time to live (default: auto)");
    senderGrid.add(tfTTL, 1, r++);
    
    senderGrid.add(new Label("Priority Level (optional)"), 0, r); 
    tfPriorityLevel.setPromptText("1-5, default: 1");
    senderGrid.add(tfPriorityLevel, 1, r++);
    
    senderGrid.add(cbUseRL, 1, r++);

        HBox sendRow = new HBox(8, new Label("Host:"), tfSendHost, new Label("Port:"), tfSendPort, btnSend);
        senderGrid.add(sendRow, 0, r, 2, 1);
    }

    private Node buildReceiverPane() {
        VBox v = new VBox(8);
        v.setPadding(new Insets(10));
        HBox h = new HBox(8, new Label("Listen port:"), tfListenPort, btnListen);
        HBox.setHgrow(tfListenPort, Priority.NEVER);
        v.getChildren().addAll(h, lvReceived, lblStatus);
        lvReceived.setPrefWidth(360);
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
                } else {
                    setText(String.format("[%s] %s -> %s : %s", item.getPacketId(), item.getSourceUserId(), item.getDestinationUserId(), item.getDecodedPayload()));
                }
            }
        });
    }
}
