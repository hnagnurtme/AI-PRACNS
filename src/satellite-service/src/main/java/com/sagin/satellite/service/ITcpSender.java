package com.sagin.satellite.service;

import com.sagin.satellite.model.Packet;

public interface ITcpSender {
    void send(Packet packet) throws Exception;
}
