package com.sagsins.core.exception;

public enum ExceptionConstants {

	INVALID_AUTHENTICATED(102, "INVALID_AUTHENTICATED"),

	INTERNAL_SERVER_ERROR(103, "INTERNAL_SERVER_ERROR"),

    NOT_FOUND_EXCEPTION(104, "NOT_FOUND_EXCEPTION"),
    
    DUPLICATE_KEY(105, "DUPLICATE_KEY"),
    ;

    private final int code;
	private final String messageName;

	ExceptionConstants(int code, String messageName) {
		this.code = code;
		this.messageName = messageName;
	}

	public int getCode() {
		return code;
	}

	public String getMessageName() {
		return messageName;
	}
}