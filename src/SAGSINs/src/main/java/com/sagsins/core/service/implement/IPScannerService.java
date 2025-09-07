package com.sagsins.core.service.implement;

import java.net.InetAddress;
import java.util.ArrayList;
import java.util.Collections;
import java.util.List;
import java.util.concurrent.*;
import java.util.concurrent.atomic.AtomicInteger;

import org.apache.commons.net.util.SubnetUtils;
import org.springframework.stereotype.Service;

import com.sagsins.core.service.IIPScanerService;
import com.sagsins.core.utils.NetworkUtils;

@Service
public class IPScannerService implements IIPScanerService {

    private static final int TIMEOUT_MS = 100;

    @Override
    public List<String> getAvailableIps(int maxResults) {
        String cidr = NetworkUtils.getLocalCidr();
        if (cidr == null) {
            throw new RuntimeException("Không thể xác định CIDR của mạng cục bộ.");
        }
        SubnetUtils subnetUtils = new SubnetUtils(cidr);
        String[] allIps = subnetUtils.getInfo().getAllAddresses();

        List<String> availableIps = Collections.synchronizedList(new ArrayList<>());
        AtomicInteger count = new AtomicInteger(0);

        ExecutorService executor = Executors.newFixedThreadPool(50);

        for (String ip : allIps) {
            executor.submit(() -> {
                if (count.get() >= maxResults)
                    return; 
                if (isAvailable(ip)) {
                    availableIps.add(ip);
                    count.incrementAndGet();
                }
            });
        }

        executor.shutdown();
        try {
            executor.awaitTermination(60, TimeUnit.SECONDS);
        } catch (InterruptedException e) {
            Thread.currentThread().interrupt();
        }

        return availableIps;
    }

    private boolean isAvailable(String ip) {
        try {
            InetAddress address = InetAddress.getByName(ip);
            return !address.isReachable(TIMEOUT_MS);
        } catch (Exception e) {
            return false;
        }
    }

}
