package com.chatapp.chatapp.model.services;

import com.chatapp.chatapp.model.interfaces.ThemeService;

import javafx.scene.Scene;

public class ThemeServiceImpl implements ThemeService {
    private final Scene scene;
    private boolean isDark = false;
    
    public ThemeServiceImpl(Scene scene) {
        this.scene = scene;
    }
    
    @Override
    public void switchToLightTheme() {
        if (scene != null) {
            scene.getStylesheets().clear();
            scene.getStylesheets().add(getClass().getResource("/css/main-style.css").toExternalForm());
            isDark = false;
        }
    }
    
    @Override
    public void switchToDarkTheme() {
        if (scene != null) {
            scene.getStylesheets().clear();
            scene.getStylesheets().add(getClass().getResource("/css/dark-theme.css").toExternalForm());
            isDark = true;
        }
    }
    
    @Override
    public void toggleTheme() {
        if (isDark) {
            switchToLightTheme();
        } else {
            switchToDarkTheme();
        }
    }
    
    @Override
    public boolean isDarkTheme() {
        return isDark;
    }
}