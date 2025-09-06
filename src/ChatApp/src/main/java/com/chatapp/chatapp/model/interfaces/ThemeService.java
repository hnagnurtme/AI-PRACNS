package com.chatapp.chatapp.model.interfaces;

public interface ThemeService {
    void switchToLightTheme();
    void switchToDarkTheme();
    void toggleTheme();
    boolean isDarkTheme();
}
