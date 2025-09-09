package com.chatapp.chatapp.model.interfaces;

public interface IThemeService {
    void switchToLightTheme();
    void switchToDarkTheme();
    void toggleTheme();
    boolean isDarkTheme();
}
