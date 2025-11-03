package com.example.service;

import com.example.model.Packet;
import com.example.util.PacketSerializerHelper;

import java.io.IOException;
import java.io.PrintWriter;
import java.net.Socket;

/**
 * Simple TCP packet sender that serializes Packet to JSON and sends to a host:port.
 * Uses PacketSerializerHelper to ensure consistent JSON serialization across the project.
 */
public class PacketSender {

    /**
     * Send a packet to the given host:port as a single-line JSON.
     * The connection is opened, the JSON is written and the socket is closed.
     * @throws IOException when socket or I/O fails
     */
    public void send(String host, int port, Packet packet) throws IOException {
        try (Socket socket = new Socket(host, port);
             PrintWriter out = new PrintWriter(socket.getOutputStream(), true)) {
            String json = PacketSerializerHelper.serialize(packet);
            if (json == null) throw new IOException("Failed to serialize packet");
            // send a single-line JSON terminated by newline so receiver can readLine()
            out.println(json);
        }
    }
}
