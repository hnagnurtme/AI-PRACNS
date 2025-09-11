package com.chatapp.chatapp.model.services;

import com.chatapp.chatapp.model.interfaces.IThemeService;

import javafx.scene.Scene;

public class ThemeService implements IThemeService {
    private final Scene scene;
    private boolean isDark = false;
    
    public ThemeService(Scene scene) {
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