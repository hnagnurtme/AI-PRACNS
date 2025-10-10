package com.sagsins.core.service.implement;

import java.io.BufferedReader;
import java.io.File;
import java.io.InputStreamReader;
import java.util.List;
import java.util.Optional;

import org.springframework.stereotype.Service;

import com.sagsins.core.DTOs.response.DockerResposne;
import com.sagsins.core.exception.DockerException;
import com.sagsins.core.model.NodeInfo;
import com.sagsins.core.service.IDockerService;

@Service
public class DockerService implements IDockerService {

    // Lệnh gốc để chạy node
    private static final String CMD = "/usr/bin/env /Library/Java/JavaVirtualMachines/temurin-17.jdk/Contents/Home/bin/java @/var/folders/dg/5bzl6y_938d3mvk1b1n_6rwm0000gp/T/cp_34cidoaxcwf93yyiirnlpd6eh.argfile com.sagin.util.SimulationMain";

    @Override
    public Optional<String> runContainerForNode(NodeInfo nodeInfo) {
        try {
            // Tạo thư mục logs nếu chưa tồn tại
            File logDir = new File("logs");
            if (!logDir.exists()) {
                logDir.mkdirs();
            }

            String logFile = "logs/" + nodeInfo.getNodeId() + ".log";
            String fullCmd = CMD + " " + nodeInfo.getNodeId() + " >> " + logFile + " 2>&1 &";

            // Chạy lệnh bất đồng bộ
            ProcessBuilder processBuilder = new ProcessBuilder();
            processBuilder.command("bash", "-c", fullCmd);
            processBuilder.start(); // không waitFor, chạy nền

            return Optional.of("Node " + nodeInfo.getNodeId() + " started asynchronously. Logs: " + logFile);
        } catch (Exception e) {
            throw new DockerException(
                    "Error while starting container for node " + nodeInfo.getNodeId() + ": " + e.getMessage());
        }
    }

    @Override
    public boolean stopAndRemoveContainer(String nodeId) {
        try {
            // Tìm PID của node đang chạy
            String findPidCmd = "pgrep -f 'com.sagin.util.SimulationMain " + nodeId + "'";
            Process findPidProcess = Runtime.getRuntime().exec(new String[] { "bash", "-c", findPidCmd });
            BufferedReader reader = new BufferedReader(new InputStreamReader(findPidProcess.getInputStream()));
            String pid = reader.readLine();
            findPidProcess.waitFor();

            if (pid != null && !pid.isEmpty()) {
                // Kill process theo PID
                String killCmd = "kill " + pid;
                Process killProcess = Runtime.getRuntime().exec(new String[] { "bash", "-c", killCmd });
                int exitCode = killProcess.waitFor();
                return exitCode == 0;
            } else {
                System.err.println("No running process found for node " + nodeId);
                return false;
            }
        } catch (Exception e) {
            System.err.println("Error while stopping container for node " + nodeId + ": " + e.getMessage());
            return false;
        }
    }

    @Override
    public Optional<String> getContainerStatus(String nodeId) {
        try {
            String findPidCmd = "pgrep -f 'com.sagin.util.SimulationMain " + nodeId + "'";
            Process findPidProcess = Runtime.getRuntime().exec(new String[] { "bash", "-c", findPidCmd });
            BufferedReader reader = new BufferedReader(new InputStreamReader(findPidProcess.getInputStream()));
            String pid = reader.readLine();
            findPidProcess.waitFor();

            if (pid != null && !pid.isEmpty()) {
                return Optional.of("Container for node " + nodeId + " is running with PID: " + pid);
            } else {
                return Optional.of("No running container found for node " + nodeId);
            }
        } catch (Exception e) {
            throw new DockerException("Error while checking status for node " + nodeId + ": " + e.getMessage());
        }
    }

    @Override
    public List<DockerResposne> getAllContainers(boolean isRunning ) {
        try {
            // Lệnh tìm tất cả process của SimulationMain
            String cmd = "pgrep -fl 'com.sagin.util.SimulationMain'";
            Process process = Runtime.getRuntime().exec(new String[] { "bash", "-c", cmd });

            BufferedReader reader = new BufferedReader(new InputStreamReader(process.getInputStream()));
            String line;
            List<DockerResposne> containers = new java.util.ArrayList<>();

            while ((line = reader.readLine()) != null) {
                // line: "<PID> com.sagin.util.SimulationMain <NODE_ID>"
                String[] parts = line.split("\\s+");
                if (parts.length >= 3) {
                    String pid = parts[0];
                    String nodeId = parts[2];
                    containers.add(new DockerResposne(nodeId, pid, true));
                }
            }

            process.waitFor();

            return containers;
        } catch (Exception e) {
            throw new DockerException("Error while listing containers: " + e.getMessage());
        }
    }
}
