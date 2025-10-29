package com.sagsins.core.DTOs.request;

import com.sagsins.core.model.Packet;

import lombok.AllArgsConstructor;
import lombok.Data;
import lombok.NoArgsConstructor;

@Data
@AllArgsConstructor
@NoArgsConstructor
public class TwoPacket {
    Packet DijkstraPacket;
    Packet RLPacket;
}
