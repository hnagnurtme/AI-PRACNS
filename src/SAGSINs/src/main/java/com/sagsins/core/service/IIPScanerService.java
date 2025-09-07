package com.sagsins.core.service;

import java.util.List;

public interface IIPScanerService {
    List<String> getAvailableIps(int maxResults);
}
