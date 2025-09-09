package com.chatapp.chatapp.controller;

import java.net.URL;
import java.util.Map;
import java.util.ResourceBundle;

import com.chatapp.auth.model.entities.UserSession;
import com.chatapp.auth.view.AuthView;
import com.chatapp.chatapp.model.interfaces.IAccessPointService;
import com.chatapp.chatapp.model.interfaces.IChatService;
import com.chatapp.chatapp.model.interfaces.ILogMonitoringService;
import com.chatapp.chatapp.model.interfaces.IMainWindowView;
import com.chatapp.chatapp.model.interfaces.INetworkTopologyService;
import com.chatapp.chatapp.model.interfaces.IStatusService;
import com.chatapp.chatapp.model.interfaces.IThemeService;
import com.chatapp.chatapp.model.services.AccessPointService;
import com.chatapp.chatapp.model.services.ChatService;
import com.chatapp.chatapp.model.services.LogMonitoringService;
import com.chatapp.chatapp.model.services.NetworkTopologyService;
import com.chatapp.chatapp.model.services.StatusService;
import com.chatapp.chatapp.model.services.ThemeService;

import javafx.application.Platform;
import javafx.fxml.FXML;
import javafx.fxml.Initializable;
import javafx.scene.control.Alert;
import javafx.scene.control.Button;
import javafx.scene.control.ButtonType;
import javafx.scene.control.ComboBox;
import javafx.scene.control.Label;
import javafx.scene.control.MenuBar;
import javafx.scene.control.MenuItem;
import javafx.scene.control.RadioButton;
import javafx.scene.control.ScrollPane;
import javafx.scene.control.SplitPane;
import javafx.scene.control.TableView;
import javafx.scene.control.TextArea;
import javafx.scene.control.TextField;
import javafx.scene.control.ToggleGroup;
import javafx.scene.layout.HBox;
import javafx.scene.layout.Pane;
import javafx.scene.layout.VBox;
import javafx.stage.Stage;

public class MainWindowController implements Initializable, IMainWindowView {

    // ===== FXML COMPONENTS =====
    
    // Menu Bar
    @FXML private MenuBar menuBar;
    @FXML private MenuItem exitMenuItem;
    @FXML private MenuItem logoutMenuItem;  // THÊM LOGOUT MENU ITEM
    @FXML private MenuItem lightThemeMenuItem;
    @FXML private MenuItem darkThemeMenuItem;

    // Status Bar
    @FXML private HBox statusBar;
    @FXML private Label connectionStatusLabel;
    @FXML private Label currentNodeLabel;
    @FXML private Label networkStatsLabel;

    // Layout Panels
    @FXML private SplitPane mainSplitPane;
    @FXML private SplitPane leftSplitPane;
    @FXML private SplitPane rightSplitPane;

    // Chat Panel Components
    @FXML private VBox chatPanel;
    @FXML private TextArea chatHistoryArea;
    @FXML private TextField messageInputField;
    @FXML private ComboBox<String> userSelectionCombo;
    @FXML private Button sendButton;
    @FXML private TextField userSearchField;
    @FXML private Button clearChatButton;

    // Topology Panel Components
    @FXML private VBox topologyPanel;
    @FXML private ScrollPane topologyScrollPane;
    @FXML private Pane topologyCanvas;
    @FXML private Button zoomInButton;
    @FXML private Button zoomOutButton;
    @FXML private Button resetViewButton;

    // Access Point Panel Components
    @FXML private VBox accessPointPanel;
    @FXML private RadioButton satelliteRadio;
    @FXML private RadioButton groundStationRadio;
    @FXML private ComboBox<String> accessPointCombo;
    @FXML private Button aiSuggestionButton;
    @FXML private Label aiRecommendationLabel;
    @FXML private Button quickConnectButton;
    @FXML private Label connectionIndicator;

    // Log Panel Components
    @FXML private VBox logPanel;
    @FXML private TableView<Object> packetLogTable;
    @FXML private TextField searchField;
    @FXML private ComboBox<String> filterCombo;
    @FXML private Button exportLogButton;

    // ===== SERVICES =====
    private IChatService chatService;
    private INetworkTopologyService topologyService;
    private IAccessPointService accessPointService;
    private ILogMonitoringService logService;
    private IThemeService themeService;
    private IStatusService statusService;

    // ===== PRIVATE FIELDS =====
    private ToggleGroup accessPointGroup;

    // ===== INITIALIZATION =====
    
    @Override
    public void initialize(URL location, ResourceBundle resources) {
        System.out.println("Initializing MainWindowController...");
        
        setupUIComponents();
        initializeServices();
        setupEventHandlers();
        setupLayout();

        System.out.println("MainWindowController initialized successfully!");
    }

    /**
     * Setup UI components that need initial configuration
     */
    private void setupUIComponents() {
        // Setup access point radio buttons
        if (satelliteRadio != null && groundStationRadio != null) {
            accessPointGroup = new ToggleGroup();
            satelliteRadio.setToggleGroup(accessPointGroup);
            groundStationRadio.setToggleGroup(accessPointGroup);
            satelliteRadio.setSelected(true);
        }
    }

    /**
     * Initialize all service implementations
     */
    private void initializeServices() {
        try {
            // Chat Service
            this.chatService = new ChatService(
                chatHistoryArea, 
                messageInputField, 
                userSelectionCombo
            );
            
            // **THÊM DÒNG NÀY** - Initialize ChatService
            if (chatService instanceof ChatService impl) {
                impl.initialize();
            }

            // Topology Service  
            this.topologyService = new NetworkTopologyService(topologyCanvas);

            // Access Point Service
            this.accessPointService = new AccessPointService(
                accessPointCombo, 
                aiRecommendationLabel, 
                connectionIndicator
            );

            // Log Monitoring Service
            this.logService = new LogMonitoringService(
                packetLogTable, 
                searchField, 
                filterCombo
            );

            // Status Service
            this.statusService = new StatusService(
                connectionStatusLabel, 
                currentNodeLabel, 
                networkStatsLabel
            );

            // Theme Service (initialize after scene is available)
            Platform.runLater(() -> {
                if (menuBar != null && menuBar.getScene() != null) {
                    this.themeService = new ThemeService(menuBar.getScene());
                }
            });

            System.out.println("All services initialized successfully");
            
        } catch (Exception e) {
            System.err.println("Error initializing services: " + e.getMessage());
            e.printStackTrace();
        }
    }

    /**
     * Setup all event handlers
     */
    private void setupEventHandlers() {
        setupMenuHandlers();
        setupChatHandlers();
        setupTopologyHandlers();
        setupAccessPointHandlers();
        setupLogHandlers();
    }

    /**
     * Setup menu event handlers
     */
    private void setupMenuHandlers() {
        if (exitMenuItem != null) {
            exitMenuItem.setOnAction(e -> Platform.exit());
        }

        // THÊM LOGOUT HANDLER
        if (logoutMenuItem != null) {
            logoutMenuItem.setOnAction(e -> handleLogout());
        }

        if (lightThemeMenuItem != null) {
            lightThemeMenuItem.setOnAction(e -> switchToLightTheme());
        }

        if (darkThemeMenuItem != null) {
            darkThemeMenuItem.setOnAction(e -> switchToDarkTheme());
        }
    }

    // THÊM LOGOUT METHODS
    @FXML
    private void handleLogout() {
        // Show confirmation dialog
        Alert confirmAlert = new Alert(Alert.AlertType.CONFIRMATION);
        confirmAlert.setTitle("Logout Confirmation");
        confirmAlert.setHeaderText("Are you sure you want to logout?");
        confirmAlert.setContentText("You will be returned to the login screen.");
        
        confirmAlert.showAndWait().ifPresent(response -> {
            if (response == ButtonType.OK) {
                performLogout();
            }
        });
    }

    private void performLogout() {
        try {
            // Clear user session
            UserSession.logout();
            
            // Close current window
            Stage currentStage = (Stage) menuBar.getScene().getWindow();
            
            // Open login window
            Stage loginStage = new Stage();
            AuthView authApp = new AuthView();
            authApp.start(loginStage);
            
            // Close main window
            currentStage.close();
            
        } catch (Exception e) {
            // Show error dialog
            Alert errorAlert = new Alert(Alert.AlertType.ERROR);
            errorAlert.setTitle("Logout Error");
            errorAlert.setHeaderText("Failed to logout");
            errorAlert.setContentText("Error: " + e.getMessage());
            errorAlert.showAndWait();
        }
    }

    /**
     * Setup chat panel event handlers
     */
    private void setupChatHandlers() {
        if (sendButton != null) {
            sendButton.setOnAction(e -> sendMessage());
        }
        
        if (messageInputField != null) {
            messageInputField.setOnAction(e -> sendMessage());
        }
        
        if (clearChatButton != null) {
            clearChatButton.setOnAction(e -> clearChat());
        }
        
        if (userSearchField != null) {
            userSearchField.setOnAction(e -> searchUser());
        }
    }

    /**
     * Setup topology panel event handlers
     */
    private void setupTopologyHandlers() {
        if (zoomInButton != null) {
            zoomInButton.setOnAction(e -> zoomInTopology());
        }
        
        if (zoomOutButton != null) {
            zoomOutButton.setOnAction(e -> zoomOutTopology());
        }
        
        if (resetViewButton != null) {
            resetViewButton.setOnAction(e -> resetTopologyView());
        }
    }

    /**
     * Setup access point panel event handlers
     */
    private void setupAccessPointHandlers() {
        if (aiSuggestionButton != null) {
            aiSuggestionButton.setOnAction(e -> getAISuggestion());
        }
        
        if (quickConnectButton != null) {
            quickConnectButton.setOnAction(e -> quickConnect());
        }
    }

    /**
     * Setup log panel event handlers
     */
    private void setupLogHandlers() {
        if (exportLogButton != null) {
            exportLogButton.setOnAction(e -> exportLogs());
        }
    }

    /**
     * Setup layout and split pane positions
     */
    private void setupLayout() {
        Platform.runLater(() -> {
            try {
                if (mainSplitPane != null) {
                    mainSplitPane.setDividerPositions(0.7); // 70% left, 30% right
                }
                
                if (leftSplitPane != null) {
                    leftSplitPane.setDividerPositions(0.6); // 60% chat, 40% topology
                }
                
                if (rightSplitPane != null) {
                    rightSplitPane.setDividerPositions(0.5); // 50% access point, 50% log
                }
                
                System.out.println("Layout configured successfully");
            } catch (Exception e) {
                System.err.println("Error setting up layout: " + e.getMessage());
            }
        });
    }

    // ===== EVENT HANDLER METHODS =====

    /**
     * Handle sending chat message
     */
    private void sendMessage() {
        try {
            if (chatService instanceof ChatService impl) {
                impl.sendCurrentMessage();
            }
        } catch (Exception e) {
            System.err.println("Error sending message: " + e.getMessage());
        }
    }

    /**
     * Switch to light theme
     */
    private void switchToLightTheme() {
        if (themeService != null) {
            themeService.switchToLightTheme();
        }
    }

    /**
     * Switch to dark theme
     */
    private void switchToDarkTheme() {
        if (themeService != null) {
            themeService.switchToDarkTheme();
        }
    }

    /**
     * Zoom in topology view
     */
    private void zoomInTopology() {
        if (topologyService != null) {
            topologyService.zoomIn();
        }
    }

    /**
     * Zoom out topology view
     */
    private void zoomOutTopology() {
        if (topologyService != null) {
            topologyService.zoomOut();
        }
    }

    /**
     * Reset topology view
     */
    private void resetTopologyView() {
        if (topologyService != null) {
            topologyService.resetView();
        }
    }

    /**
     * Get AI access point suggestion
     */
    private void getAISuggestion() {
        if (accessPointService != null) {
            accessPointService.getAIRecommendation();
        }
    }

    /**
     * Quick connect to selected access point
     */
    private void quickConnect() {
        if (accessPointService instanceof AccessPointService impl) {
            impl.quickConnect();
        }
    }

    /**
     * Export logs
     */
    private void exportLogs() {
        if (logService instanceof LogMonitoringService impl) {
            impl.exportLogAction();
        }
    }

    private void searchUser() {
        if (userSearchField != null && chatService instanceof ChatService impl) {
            String email = userSearchField.getText().trim();
            if (!email.isEmpty()) {
                impl.searchAndAddUser(email);
                userSearchField.clear();
            }
        }
    }

    private void clearChat() {
        if (chatService != null) {
            chatService.clearChatHistory();
        }
    }

    // ===== MAINWINDOWVIEW INTERFACE IMPLEMENTATION =====

    @Override
    public IChatService getChatService() {
        return chatService;
    }

    @Override
    public INetworkTopologyService getTopologyService() {
        return topologyService;
    }

    @Override
    public IAccessPointService getAccessPointService() {
        return accessPointService;
    }

    @Override
    public ILogMonitoringService getLogService() {
        return logService;
    }

    @Override
    public IThemeService getThemeService() {
        return themeService;
    }

    @Override
    public IStatusService getStatusService() {
        return statusService;
    }

    @Override
    public void initializeView() {
        // Initialize view components here if needed
    }

    @Override
    public void shutdown() {
        try {
            System.out.println("MainWindowController shutting down...");
            
            // Cleanup resources if needed
            if (chatService != null) {
                // Cleanup chat service if needed
            }
            
            if (topologyService != null) {
                // Cleanup topology service if needed
            }
            
            if (accessPointService != null) {
                // Cleanup access point service if needed
            }
            
            if (logService != null) {
                // Cleanup log service if needed
            }
            
            if (themeService != null) {
                // Cleanup theme service if needed
            }
            
            if (statusService != null) {
                // Cleanup status service if needed
            }
            
            if (chatService instanceof ChatService impl) {
                impl.shutdown();
            }
            
            System.out.println("MainWindowController shutdown completed");
            
        } catch (Exception e) {
            System.err.println("Error during shutdown: " + e.getMessage());
        }
    }

    // ===== PUBLIC UTILITY METHODS =====

    /**
     * Update connection status in status bar
     */
    public void updateConnectionStatus(String status) {
        if (statusService != null) {
            statusService.updateConnectionStatus(status);
        }
    }

    /**
     * Update current node in status bar
     */
    public void updateCurrentNode(String node) {
        if (statusService != null) {
            statusService.updateCurrentNode(node);
        }
    }

    /**
     * Update network stats in status bar
     */
    public void updateNetworkStats(String stats) {
        if (statusService != null) {
            statusService.updateNetworkStats(stats);
        }
    }

    /**
     * Get current toggle group for access points
     */
    public ToggleGroup getAccessPointGroup() {
        return accessPointGroup;
    }
}
