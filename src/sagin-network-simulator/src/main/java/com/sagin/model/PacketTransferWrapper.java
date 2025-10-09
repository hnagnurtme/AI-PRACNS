package com.sagin.model;

import java.io.Serializable;
import lombok.*;

/**
 * Đối tượng bao bọc dữ liệu để truyền qua RPC (Socket)
 */
@Getter
@Setter
@NoArgsConstructor
@AllArgsConstructor
public class PacketTransferWrapper implements Serializable {
    private static final long serialVersionUID = 1L;
    
    private Packet packet;
    private LinkMetric linkMetric;
}