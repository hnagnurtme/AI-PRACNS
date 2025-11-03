package com.example.service;

import com.example.model.Packet;
import com.fasterxml.jackson.databind.ObjectMapper;

import java.io.BufferedReader;
import java.io.IOException;
import java.io.InputStreamReader;
import java.net.ServerSocket;
import java.net.Socket;
import java.util.concurrent.ExecutorService;
import java.util.concurrent.Executors;
import java.util.function.Consumer;

/**
 * Simple TCP packet receiver. Runs a ServerSocket in a background thread and
 * calls the provided Consumer<Packet> whenever a JSON packet arrives.
 *
 * Note: This implementation expects each incoming Packet JSON to be sent as one line
 * (sender writes json + "\n"). It's sufficient for small demo/testing purposes.
 */
public class PacketReceiver {

    private final ObjectMapper mapper = new ObjectMapper();
    private ServerSocket serverSocket;
    private ExecutorService acceptor = Executors.newSingleThreadExecutor();
    private volatile boolean running = false;

    /**
     * Start listening on the given port. Received packets are delivered to packetConsumer.
     */
    public void start(int port, Consumer<Packet> packetConsumer) throws IOException {
        if (running) return;
        serverSocket = new ServerSocket(port);
        running = true;

        acceptor.submit(() -> {
            while (running && !serverSocket.isClosed()) {
                try {
                    Socket client = serverSocket.accept();
                    // handle client on a short-lived task
                    handleClient(client, packetConsumer);
                } catch (IOException e) {
                    if (running) {
                        e.printStackTrace();
                    }
                }
            }
        });
    }

    private void handleClient(Socket client, Consumer<Packet> packetConsumer) {
        Executors.newSingleThreadExecutor().submit(() -> {
            try (BufferedReader in = new BufferedReader(new InputStreamReader(client.getInputStream()))) {
                String line = in.readLine();
                if (line != null && !line.isEmpty()) {
                    try {
                        Packet p = mapper.readValue(line, Packet.class);
                        packetConsumer.accept(p);
                    } catch (Exception ex) {
                        ex.printStackTrace();
                    }
                }
            } catch (IOException e) {
                e.printStackTrace();
            } finally {
                try { client.close(); } catch (IOException ignored) {}
            }
        });
    }

    /**
     * Stop listening and shut down background threads.
     */
    public void stop() {
        running = false;
        try {
            if (serverSocket != null) serverSocket.close();
        } catch (IOException ignored) {}
        acceptor.shutdownNow();
    }
}
