package com.sagsins.core.repository;

import java.util.List;

import com.sagsins.core.model.IPAddress;

public interface INetworkRepository {
    List<IPAddress> getAllIPs();
}
