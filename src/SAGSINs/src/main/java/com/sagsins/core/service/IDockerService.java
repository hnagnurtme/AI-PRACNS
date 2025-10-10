package com.sagsins.core.service;

import com.sagsins.core.DTOs.response.DockerResposne;
import com.sagsins.core.model.NodeInfo;

import java.util.List;
import java.util.Optional;

/**
 * Định nghĩa các nghiệp vụ liên quan đến việc quản lý Docker/Container cho các Node.
 */
public interface IDockerService {

    /**
     * Tự động tạo và khởi chạy một Container Docker cho Node mới được tạo.
     * Container ID sẽ được lưu trữ trong NodeInfo (nếu cần).
     *
     * @param nodeInfo Dữ liệu của NodeInfo cần triển khai container.
     * @return Container ID nếu triển khai thành công, Optional rỗng nếu thất bại.
     */
    Optional<String> runContainerForNode(NodeInfo nodeInfo);

    /**
     * Dừng và xóa Container Docker liên kết với Node ID.
     *
     * @param nodeId ID của Node cần dừng container.
     * @return true nếu Container được dừng/xóa thành công.
     */
    boolean stopAndRemoveContainer(String nodeId);

    /**
     * Lấy ID/Trạng thái của Container dựa trên Node ID.
     *
     * @param nodeId ID của Node.
     * @return Container ID.
     */
    Optional<String> getContainerStatus(String nodeId);



    // get all conatiner 
    List<DockerResposne> getAllContainers(boolean isRuunning);
}