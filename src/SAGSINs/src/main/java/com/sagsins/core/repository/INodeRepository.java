package com.sagsins.core.repository;

import com.sagsins.core.model.NodeInfo;

import org.springframework.data.mongodb.repository.MongoRepository;
import org.springframework.stereotype.Repository;

/**
 * Interface Repository định nghĩa các thao tác CRUD cơ bản cho NodeInfo.
 * Đây là giao diện chuẩn hóa tầng truy cập dữ liệu (Data Access Layer).
 */
@Repository
public interface INodeRepository extends MongoRepository<NodeInfo, String>   {

}