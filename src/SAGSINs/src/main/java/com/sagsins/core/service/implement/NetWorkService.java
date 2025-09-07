package com.sagsins.core.service.implement;

import java.util.List;

import org.springframework.stereotype.Service;

import com.sagsins.core.model.IPAddress;
import com.sagsins.core.repository.INetworkRepository;
import com.sagsins.core.service.INetworkService;

@Service
public class NetWorkService implements INetworkService{
    private final INetworkRepository networkRepository;

    public NetWorkService(INetworkRepository networkRepository) {
        this.networkRepository = networkRepository;
    }

    @Override
    public List<IPAddress> getAllIPs() {
        return networkRepository.getAllIPs();
    }
}
