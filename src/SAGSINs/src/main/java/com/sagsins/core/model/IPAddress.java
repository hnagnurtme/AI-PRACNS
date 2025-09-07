package com.sagsins.core.model;

import lombok.*;

@Getter
@Setter
@NoArgsConstructor
@AllArgsConstructor

public class IPAddress {
    private String ip;
    private boolean isIPv4;
    private boolean isValid;
    private Integer port;
}
