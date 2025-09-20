package com.chatapp.chatapp.model.interfaces;

import java.util.concurrent.ExecutionException;

import com.chatapp.auth.model.entities.User;
import com.chatapp.chatapp.model.entities.Node;

public interface INodeService {
    public Node findNearestNode(User user) throws ExecutionException, InterruptedException;
    double haversine(double lat1, double lon1, double lat2, double lon2);
}
