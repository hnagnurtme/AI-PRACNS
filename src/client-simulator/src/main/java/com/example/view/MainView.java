package com.example.view;

import com.example.model.Packet;
import com.example.model.ServiceType; // Import
import javafx.collections.ObservableList;
import javafx.geometry.Insets;
import javafx.geometry.Pos;
import javafx.scene.Node;
import javafx.scene.control.*;
import javafx.scene.layout.*;

/**
 * Modern redesigned view component, optimized for inline styling and performance.
 */
public class MainView {

    // === 1. DỌN DẸP CÁC TRƯỜNG KHÔNG SỬ DỤNG ===
    // Chỉ giữ lại các trường thực sự được thêm vào senderGrid

    // Sender controls
    public GridPane senderGrid;
    public ComboBox<String> cbSenderUsername = new ComboBox<>();
    public ComboBox<String> cbDestinationUsername = new ComboBox<>();
    public TextArea taPayload = new TextArea();
    public TextField tfPayloadSizeByte = new TextField();
    public ComboBox<ServiceType> cbServiceType = new ComboBox<>();
    public Label lblQoSDetail = new Label();
    public TextField tfPacketCount = new TextField("1");
    public Button btnSend = new Button("Send Packet");
    
    // Receiver controls
    public TextField tfListenPort = new TextField("9000"); // Giữ lại vì nó có giá trị default
    public Button btnListen = new Button("Start Listening");
    public Button btnClearLog = new Button("Clear Log");
    public ListView<Packet> lvReceived = new ListView<>();
    public Label lblStatus = new Label("Ready");

    public BorderPane root = new BorderPane();

    // === 2. GOM STYLE VÀO HẰNG SỐ (CONSTANTS) ===
    
    // Color scheme
    private static final String PRIMARY_COLOR = "#2c3e50";
    private static final String ACCENT_COLOR = "#3498db";
    private static final String ACCENT_HOVER = "#2980b9";
    private static final String DANGER_COLOR = "#e74c3c";
    private static final String DANGER_HOVER = "#c0392b";
    private static final String BACKGROUND_COLOR = "#ecf0f1";
    private static final String CARD_COLOR = "#ffffff";
    private static final String BORDER_COLOR = "#ced4da";
    private static final String BORDER_LIGHT = "#dee2e6";

    // Style strings for Text Inputs (TextField, TextArea)
    private static final String STYLE_TEXT_DEFAULT = String.join(";",
        "-fx-background-color: white",
        "-fx-border-color: " + BORDER_COLOR,
        "-fx-border-radius: 5",
        "-fx-background-radius: 5",
        "-fx-padding: 8",
        "-fx-font-size: 13px"
    );
    private static final String STYLE_TEXT_FOCUSED = String.join(";",
        "-fx-background-color: white",
        "-fx-border-color: " + ACCENT_COLOR,
        "-fx-border-width: 2",
        "-fx-border-radius: 5",
        "-fx-background-radius: 5",
        "-fx-padding: 8",
        "-fx-font-size: 13px"
    );

    // Style strings for Buttons
    private static final String STYLE_BTN_PRIMARY_DEFAULT = String.join(";",
        "-fx-background-color: " + ACCENT_COLOR,
        "-fx-text-fill: white",
        "-fx-font-size: 14px",
        "-fx-font-weight: bold",
        "-fx-background-radius: 5",
        "-fx-padding: 10 20",
        "-fx-cursor: hand"
    );
    private static final String STYLE_BTN_PRIMARY_HOVER = String.join(";",
        "-fx-background-color: " + ACCENT_HOVER,
        "-fx-text-fill: white",
        "-fx-font-size: 14px",
        "-fx-font-weight: bold",
        "-fx-background-radius: 5",
        "-fx-padding: 10 20",
        "-fx-cursor: hand",
        "-fx-effect: dropshadow(gaussian, rgba(0,0,0,0.3), 5, 0, 0, 2)"
    );
    private static final String STYLE_BTN_DANGER_DEFAULT = String.join(";",
        "-fx-background-color: " + DANGER_COLOR,
        "-fx-text-fill: white",
        "-fx-font-size: 14px",
        "-fx-font-weight: bold",
        "-fx-background-radius: 5",
        "-fx-padding: 10 20",
        "-fx-cursor: hand"
    );
    private static final String STYLE_BTN_DANGER_HOVER = String.join(";",
        "-fx-background-color: " + DANGER_HOVER,
        "-fx-text-fill: white",
        "-fx-font-size: 14px",
        "-fx-font-weight: bold",
        "-fx-background-radius: 5",
        "-fx-padding: 10 20",
        "-fx-cursor: hand",
        "-fx-effect: dropshadow(gaussian, rgba(0,0,0,0.3), 5, 0, 0, 2)"
    );

    public MainView(ObservableList<Packet> receivedItems) {
        // applyModernStyles(); // Hàm này rỗng, có thể xóa
        buildSender();
        buildReceiver(receivedItems);

        SplitPane split = new SplitPane();
        
        VBox senderContainer = createStyledPanel("📤 Packet Sender", new ScrollPane(senderGrid));
        VBox receiverContainer = createStyledPanel("📥 Packet Receiver", buildReceiverPane());
        
        split.getItems().addAll(senderContainer, receiverContainer);
        split.setDividerPositions(0.35);
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
        VBox.setVgrow(panel, Priority.ALWAYS);
        
        return wrapper;
    }

    private Separator createDivider() {
        Separator sep = new Separator();
        sep.setStyle("-fx-background-color: " + ACCENT_COLOR + ";");
        return sep;
    }

    // private void applyModernStyles() {} // XÓA VÌ HÀM RỖNG

    private void buildSender() {
        senderGrid = new GridPane();
        senderGrid.setHgap(15);
        senderGrid.setVgap(12);
        senderGrid.setPadding(new Insets(15));

        int r = 0;
        
        addSectionHeader(senderGrid, "👥 User Information", r++);
        
        addStyledField(senderGrid, "Sender Username", cbSenderUsername, r++, 
            "Select or type sender username");
        cbSenderUsername.setEditable(true);
        
        addStyledField(senderGrid, "Destination Username", cbDestinationUsername, r++,
            "Select or type destination username");
        cbDestinationUsername.setEditable(true);
        
        r++;
        addSectionHeader(senderGrid, "✉️ Message Content", r++);
        
        senderGrid.add(createStyledLabel("Payload"), 0, r);
        taPayload.setPrefRowCount(4);
        taPayload.setPromptText("Enter your message here...");
        applyTextInputStyles(taPayload); // Dùng hàm tối ưu
        senderGrid.add(taPayload, 1, r++);
        
        addStyledField(senderGrid, "Packet Size (bytes)", tfPayloadSizeByte, r++,
            "Auto-calculated or enter manually");
        
        r++;
        addSectionHeader(senderGrid, "⚙️ Service Configuration", r++);
        
        senderGrid.add(createStyledLabel("Service Type"), 0, r);
        cbServiceType.getItems().addAll(com.example.model.ServiceType.values());
        styleComboBox(cbServiceType); // Vẫn dùng hàm riêng cho ComboBox
        cbServiceType.setPromptText("Select service quality type");
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
            "-fx-border-color: " + BORDER_LIGHT + ";" +
            "-fx-border-radius: 5;"
        );
        senderGrid.add(lblQoSDetail, 1, r++);
        
        r++;
        addSectionHeader(senderGrid, "🎯 Sending Options", r++);
        
        addStyledField(senderGrid, "Number of Packet Pairs", tfPacketCount, r++,
            "Each pair = 1 RL + 1 non-RL (default: 1)");
        
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
        
        r++;
        
        stylePrimaryButton(btnSend); // Dùng hàm tối ưu
        btnSend.setPrefWidth(280);
        senderGrid.add(btnSend, 1, r);
    }

    private Node buildReceiverPane() {
        VBox v = new VBox(15);
        v.setPadding(new Insets(15));
        v.setFillWidth(true);
        
        HBox controlBox = new HBox(10);
        controlBox.setAlignment(Pos.CENTER_LEFT);
        controlBox.setPadding(new Insets(10));
        controlBox.setStyle(
            "-fx-background-color: #f8f9fa;" +
            "-fx-background-radius: 5;" +
            "-fx-border-color: " + BORDER_LIGHT + ";" +
            "-fx-border-radius: 5;"
        );
        
        stylePrimaryButton(btnListen); // Dùng hàm tối ưu
        btnListen.setPrefWidth(200);
        
        styleDangerButton(btnClearLog); // Dùng hàm tối ưu MỚI
        btnClearLog.setPrefWidth(150);
        
        // XÓA CÁC LISTENER `onMouseEntered/Exited` cho btnClearLog
        
        controlBox.getChildren().addAll(btnListen, btnClearLog);
        
        Label listLabel = createStyledLabel("📨 Received Packets");
        listLabel.setStyle("-fx-font-size: 14px; -fx-font-weight: bold;");
        
        lvReceived.setStyle(
            "-fx-background-color: white;" +
            "-fx-border-color: " + BORDER_LIGHT + ";" +
            "-fx-border-radius: 5;" +
            "-fx-background-radius: 5;"
        );
        lvReceived.setPrefHeight(400);
        lvReceived.setMaxHeight(Double.MAX_VALUE);
        
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

    // === 3. TỐI ƯU HÓA LISTVIEW CELLFACTORY ===
    private void buildReceiver(ObservableList<Packet> receivedItems) {
        lvReceived.setItems(receivedItems);
        // Dùng ViewHolder Pattern để không tạo Node mới mỗi khi cuộn
        lvReceived.setCellFactory(list -> new PacketCell());
    }

    /**
     * Lớp nội bộ (inner class) theo ViewHolder pattern.
     * Nó tạo các Node MỘT LẦN và chỉ cập nhật nội dung.
     */
    private static class PacketCell extends ListCell<Packet> {
        private final VBox cell;
        private final Label badge;
        private final Label header;
        private final Label route;
        private final Label payload;
        private final HBox headerRow;
        
        // Style strings cho CellFactory
        private static final String STYLE_CELL_RL = String.join(";",
            "-fx-background-color: linear-gradient(to right, #e8f5e9, #c8e6c9)",
            "-fx-background-radius: 8",
            "-fx-border-color: #4caf50",
            "-fx-border-width: 2",
            "-fx-border-radius: 8",
            "-fx-effect: dropshadow(gaussian, rgba(76, 175, 80, 0.3), 5, 0, 0, 1)"
        );
        private static final String STYLE_CELL_STATIC = String.join(";",
            "-fx-background-color: linear-gradient(to right, #fff3e0, #ffe0b2)",
            "-fx-background-radius: 8",
            "-fx-border-color: #ff9800",
            "-fx-border-width: 2",
            "-fx-border-radius: 8",
            "-fx-effect: dropshadow(gaussian, rgba(255, 152, 0, 0.3), 5, 0, 0, 1)"
        );
        private static final String STYLE_BADGE_RL = "-fx-background-color: #4caf50; -fx-text-fill: white;";
        private static final String STYLE_BADGE_STATIC = "-fx-background-color: #ff9800; -fx-text-fill: white;";
        private static final String STYLE_HEADER_RL = "-fx-text-fill: #2e7d32;";
        private static final String STYLE_HEADER_STATIC = "-fx-text-fill: #e65100;";

        public PacketCell() {
            // Khởi tạo các Node MỘT LẦN
            badge = new Label();
            badge.setStyle(
                "-fx-font-size: 10px;" +
                "-fx-font-weight: bold;" +
                "-fx-padding: 3 8;" +
                "-fx-background-radius: 3;"
            );
            
            header = new Label();
            header.setStyle(
                "-fx-font-weight: bold;" +
                "-fx-font-size: 13px;"
            );
            
            route = new Label();
            route.setStyle("-fx-font-size: 11px; -fx-text-fill: #555;");
            
            payload = new Label();
            payload.setStyle("-fx-font-size: 12px; -fx-text-fill: #333;");
            payload.setWrapText(true);
            
            headerRow = new HBox(10, badge, header);
            headerRow.setAlignment(Pos.CENTER_LEFT);
            
            cell = new VBox(5, headerRow, route, payload);
            cell.setPadding(new Insets(10));
            
            // Set style chung cho ListCell
            setStyle("-fx-padding: 5; -fx-background-color: transparent;");
        }
        
        @Override
        protected void updateItem(Packet item, boolean empty) {
            super.updateItem(item, empty);
            if (empty || item == null) {
                setText(null);
                setGraphic(null);
            } else {
                boolean usesRL = item.isUseRL();
                
                // Chỉ cập nhật nội dung và style, KHÔNG tạo Node mới
                badge.setText(usesRL ? "🤖 RL ROUTING" : "📍 STATIC ROUTING");
                header.setText(String.format("📦 Packet ID: %s", item.getPacketId()));
                route.setText(String.format("🔀 %s → %s", 
                    item.getSourceUserId(), item.getDestinationUserId()));
                payload.setText(String.format("💬 %s", item.getDecodedPayload()));
                
                // Cập nhật style
                if (usesRL) {
                    cell.setStyle(STYLE_CELL_RL);
                    badge.setStyle(badge.getStyle() + STYLE_BADGE_RL);
                    header.setStyle(header.getStyle() + STYLE_HEADER_RL);
                } else {
                    cell.setStyle(STYLE_CELL_STATIC);
                    badge.setStyle(badge.getStyle() + STYLE_BADGE_STATIC);
                    header.setStyle(header.getStyle() + STYLE_HEADER_STATIC);
                }
                
                setText(null);
                setGraphic(cell);
            }
        }
    }


    // === STYLING HELPER METHODS (ĐÃ TỐI ƯU) ===

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

    // Hàm chung cho TextField
    private void addStyledField(GridPane grid, String labelText, TextField field, int row, String prompt) {
        grid.add(createStyledLabel(labelText), 0, row);
        field.setPromptText(prompt);
        applyTextInputStyles(field); // Dùng hàm tối ưu
        grid.add(field, 1, row);
    }
    
    // Hàm chung cho ComboBox
    private void addStyledField(GridPane grid, String labelText, ComboBox<?> combo, int row, String prompt) {
        grid.add(createStyledLabel(labelText), 0, row);
        combo.setPromptText(prompt);
        styleComboBox(combo); // Giữ nguyên
        grid.add(combo, 1, row);
    }

    /**
     * (HÀM TỐI ƯU) Áp dụng style và listener cho cả TextField và TextArea
     */
    private void applyTextInputStyles(Region input) {
        // Áp dụng cho cả TextField (con của Region) và TextArea (con của Region)
        input.setStyle(STYLE_TEXT_DEFAULT);
        input.setPrefWidth(280);
        
        // Thêm listener CHUNG
        input.focusedProperty().addListener((obs, oldVal, newVal) -> {
            input.setStyle(newVal ? STYLE_TEXT_FOCUSED : STYLE_TEXT_DEFAULT);
        });
    }
    
    // XÓA HÀM `styleTextField` và `styleTextArea` VÌ ĐÃ GOM VÀO `applyTextInputStyles`
    
    private void styleComboBox(ComboBox<?> combo) {
        combo.setStyle(
            "-fx-background-color: white;" +
            "-fx-border-color: " + BORDER_COLOR + ";" +
            "-fx-border-radius: 5;" +
            "-fx-background-radius: 5;" +
            "-fx-padding: 5;" + // ComboBox padding khác
            "-fx-font-size: 13px;"
        );
        combo.setPrefWidth(280);
        // ComboBox không cần/khó style focus listener, nên giữ riêng
    }

    /**
     * (HÀM TỐI ƯU CHUNG) Thêm hiệu ứng hover cho bất kỳ nút nào
     */
    private void applyButtonHoverEffect(Button btn, String styleDefault, String styleHover) {
        btn.setStyle(styleDefault);
        btn.setOnMouseEntered(e -> btn.setStyle(styleHover));
        btn.setOnMouseExited(e -> btn.setStyle(styleDefault));
    }

    /**
     * (HÀM TỐI ƯU) Style cho nút chính
     */
    private void stylePrimaryButton(Button btn) {
        applyButtonHoverEffect(btn, STYLE_BTN_PRIMARY_DEFAULT, STYLE_BTN_PRIMARY_HOVER);
    }
    
    /**
     * (HÀM TỐI ƯU MỚI) Style cho nút nguy hiểm
     */
    private void styleDangerButton(Button btn) {
        applyButtonHoverEffect(btn, STYLE_BTN_DANGER_DEFAULT, STYLE_BTN_DANGER_HOVER);
    }
}