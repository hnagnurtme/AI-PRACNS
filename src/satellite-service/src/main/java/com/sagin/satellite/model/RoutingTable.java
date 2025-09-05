package com.sagin.satellite.model;

import java.util.Map;
import java.util.concurrent.ConcurrentHashMap;

public class RoutingTable {
    private final Map<String, String> routes = new ConcurrentHashMap<>();

    public void addRoute(String dst, String nextHop) {
        if (dst == null || nextHop == null) return;
        routes.put(dst, nextHop);
    }

    public String getNextHop(String dst) {
        return routes.get(dst);
    }

    public void removeRoute(String dst) {
        routes.remove(dst);
    }
}
