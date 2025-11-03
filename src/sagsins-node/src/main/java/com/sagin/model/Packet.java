package com.sagin.model;

import com.fasterxml.jackson.annotation.JsonIgnore;
import com.fasterxml.jackson.annotation.JsonInclude;
import lombok.AllArgsConstructor;
import lombok.Data;
import lombok.NoArgsConstructor;

import java.nio.charset.StandardCharsets;
import java.util.ArrayList;
import java.util.Base64;
import java.util.List;

@Data 
@NoArgsConstructor
@AllArgsConstructor
@JsonInclude(JsonInclude.Include.NON_NULL) 
public class Packet {

    private String packetId;
    private String sourceUserId;
    private String destinationUserId;
    private String stationSource;
    private String stationDest;
    private String type;
    private String acknowledgedPacketId;
    private long timeSentFromSourceMs;
    private String payloadDataBase64;
    private int payloadSizeByte;
    private ServiceQoS serviceQoS;
    private int TTL;
    private String currentHoldingNodeId;
    private String nextHopNodeId;
    private List<String> pathHistory = new ArrayList<>();
    private List<HopRecord> hopRecords = new ArrayList<>();
    private double accumulatedDelayMs = 0.0;
    private int priorityLevel = 1;
    private boolean isUseRL = false;
    private double maxAcceptableLatencyMs = 150.0;
    private double maxAcceptableLossRate = 0.01;
    private boolean dropped = false;
    private String dropReason;
    private AnalysisData analysisData;

    /**
     * Giải mã payload từ Base64 sang chuỗi UTF-8.
     * @return Chuỗi đã giải mã hoặc thông báo lỗi.
     */
    @JsonIgnore 
    public String getDecodedPayload() {
        try {
            byte[] decodedBytes = Base64.getDecoder().decode(this.payloadDataBase64);
            return new String(decodedBytes, StandardCharsets.UTF_8);
        } catch (Exception e) {
            return "[Error decoding Base64]";
        }
    }
}