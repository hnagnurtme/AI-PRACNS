package com.sagin.satellite.service;

import com.sagin.satellite.model.Packet;
import com.sagin.satellite.common.SatelliteException;

import java.util.List;

/**
 * IBufferManager định nghĩa các phương thức quản lý buffer cho vệ tinh.
 * Thread-safe, hỗ trợ FIFO, kiểm tra capacity, và lấy snapshot buffer hiện tại.
 */
public interface IBufferManager {

    /**
     * Thêm packet vào buffer
     *
     * @param packet Packet cần thêm
     * @throws SatelliteException.InvalidPacketException nếu packet null hoặc không hợp lệ
     */
    void add(Packet packet) throws SatelliteException.InvalidPacketException;

    /**
     * Lấy và loại bỏ packet tiếp theo theo cơ chế FIFO
     *
     * @return Packet tiếp theo
     * @throws SatelliteException.BufferEmptyException nếu buffer rỗng
     */
    Packet poll() throws SatelliteException.BufferEmptyException;

    /**
     * Lấy snapshot tất cả packet hiện có trong buffer
     *
     * @return Danh sách packet
     */
    List<Packet> getAll();

    /**
     * Kiểm tra buffer còn capacity không
     *
     * @return true nếu còn chỗ, false nếu full
     */
    boolean hasCapacity();

    /**
     * Lấy kích thước buffer hiện tại
     *
     * @return số lượng packet
     */
    int size();

    /**
     * Xóa tất cả packet khỏi buffer
     */
    void clear();

    /**
     * Kiểm tra buffer có đầy không
     *
     * @return true nếu full, false nếu còn chỗ
     */
    default boolean isFull() {
        return !hasCapacity();
    }
}
