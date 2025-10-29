package com.sagsins.core.DTOs.request;

import java.util.List;

import lombok.AllArgsConstructor;
import lombok.Data;
import lombok.NoArgsConstructor;

@Data
@AllArgsConstructor
@NoArgsConstructor
public class BatchPacket {
    private String batchId;
    private int totalPairPackets;
    private List<TwoPacket> packets;
}
