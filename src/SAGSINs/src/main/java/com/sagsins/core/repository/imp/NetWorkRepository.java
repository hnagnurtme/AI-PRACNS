package com.sagsins.core.repository.imp;

import java.util.List;
import java.util.concurrent.ExecutionException;
import java.util.stream.StreamSupport;

import org.springframework.stereotype.Repository;

import com.google.cloud.firestore.DocumentReference;
import com.google.cloud.firestore.Firestore;
import com.sagsins.core.model.IPAddress;
import com.sagsins.core.repository.INetworkRepository;

@Repository
public class NetWorkRepository implements INetworkRepository{
    private final Firestore firestore;

    public NetWorkRepository(Firestore firestore) {
        this.firestore = firestore;
    }

    @Override
    public List<IPAddress> getAllIPs() {
        try {
            Iterable<DocumentReference> docs = firestore.collection("networks").listDocuments();
            return StreamSupport.stream(docs.spliterator(), false) 
                    .map(docRef -> {
                        try {
                            return docRef.get().get().toObject(IPAddress.class);
                        } catch (InterruptedException e) {  
                            Thread.currentThread().interrupt();
                            e.printStackTrace();
                        } catch (ExecutionException e) {
                            e.printStackTrace();
                        }
                        return null;
                    })
                    .filter(ip -> ip != null)
                    .toList();
        } catch (Exception e) {
            e.printStackTrace();
            return List.of();  
        }
    }
}
