package com.sagin.util;

/**
 * Chứa tất cả các hằng số mô phỏng (Simulation Constants) cho mạng SAGSIN.
 * <p>
 * Class này được thiết kế để không thể khởi tạo (private constructor).
 * Tất cả các giá trị đều là {@code public static final} để có thể truy cập
 * trực tiếp một cách an toàn từ bất kỳ đâu trong ứng dụng.
 * (ví dụ: {@code SimulationConstants.MIN_BATTERY}).
 * <p>
 * Các hằng số được nhóm theo chức năng để dễ dàng quản lý và tinh chỉnh.
 */
public final class SimulationConstants { // Thêm 'final' để ngăn kế thừa

    /**
     * Private constructor để ngăn chặn việc khởi tạo class tiện ích này.
     */
    private SimulationConstants() {
    }

    // ===========================================
    // ⚙️ NĂNG LƯỢNG (PIN) - ĐƠN VỊ: % PIN
    // ===========================================

    /**
     * % pin tiêu thụ để *NHẬN* một packet (Receive - RX).
     * Đây là chi phí cố định cho mỗi packet nhận được.
     */
    public static final double RX_COST_PER_PACKET = 0.0001;

    /**
     * % pin tiêu thụ để *XỬ LÝ* một packet dữ liệu thông thường (CPU).
     * Chi phí này đại diện cho việc đọc header, tra cứu bảng định tuyến, v.v.
     */
    public static final double BASE_CPU_DRAIN_COST = 0.0002;

    /**
     * % pin tiêu thụ để *XỬ LÝ* một packet yêu cầu chạy thuật toán RL (CPU-intensive).
     * Chi phí này cao hơn đáng kể so với xử lý thông thường.
     */
    public static final double RL_CPU_DRAIN_COST = 0.0015; // Ví dụ: Gấp ~7.5 lần xử lý thường

    /**
     * % pin tiêu thụ để *TRUYỀN* (Transmit - TX) cho mỗi byte dữ liệu.
     * Đây là một trong những chi phí tốn kém nhất.
     */
    public static final double TX_COST_PER_BYTE = 0.0000003;

    // ===========================================
    // ⏱ HIỆU NĂNG (ĐỘ TRỄ) - ĐƠN VỊ: Milliseconds (ms)
    // ===========================================

    /**
     * Độ trễ xử lý (CPU processing delay) cho packet dữ liệu thông thường (ms).
     */
    public static final double DATA_PROCESSING_DELAY_MS = 0.2;

    /**
     * Độ trễ xử lý (CPU processing delay) cho packet yêu cầu chạy RL (ms).
     */
    public static final double RL_PROCESSING_DELAY_MS = 2.5;

    /**
     * Độ trễ hàng đợi (queuing delay) tối đa (ms) khi buffer đầy 100%.
     * Dùng để ước tính độ trễ hàng đợi dựa trên % buffer đang sử dụng.
     */
    public static final double MAX_QUEUING_DELAY_MS = 10.0;

    /**
     * Độ dài của một "time slot" mô phỏng (ms).
     * Dùng để tính toán utilization (% thời gian bận rộn trong 1 slot).
     */
    public static final double SIMULATION_TIMESLOT_MS = 100.0;

    // ===========================================
    // 📈 HỆ SỐ MÔ PHỎNG (EMA - Exponential Moving Average)
    // ===========================================

    /**
     * Hệ số Alpha (làm mượt) cho Exponential Moving Average (EMA) của Resource Utilization.
     * Giá trị nhỏ (gần 0) -> chỉ số thay đổi chậm (ưu tiên lịch sử).
     * Giá trị lớn (gần 1) -> chỉ số thay đổi nhanh (ưu tiên giá trị tức thời).
     */
    public static final double ALPHA_UTIL = 0.05;

    /**
     * Hệ số Beta (làm mượt) cho Exponential Moving Average (EMA) của Packet Loss Rate.
     * Giá trị lớn (gần 1) -> chỉ số thay đổi chậm (ưu tiên lịch sử).
     */
    public static final double BETA_LOSS = 0.8;

    // ===========================================
    // 🌍 VẬT LÝ & MÔI TRƯỜNG
    // ===========================================

    /**
     * Hằng số chia để đổi (km) sang (ms) cho tốc độ truyền sóng (gần tốc độ ánh sáng, c ~ 300,000 km/s).
     * (300,000 km/s = 300 km/ms)
     */
    public static final double PROPAGATION_DIVISOR_KM_MS = 300.0;

    /**
     * Hằng số chuẩn hóa độ cao (km) cho việc tính tiêu thụ pin.
     * (Ví dụ: dựa trên quỹ đạo GEO ~35,786 km, làm tròn lên 40,000).
     */
    public static final double ALTITUDE_DRAIN_NORMALIZATION_KM = 40000.0;

    /**
     * Hệ số chuyển đổi từ dB (attenuation) sang hệ số nhân ảnh hưởng độ trễ.
     * Dùng trong {@code computeDetailedDelay}.
     */
    public static final double WEATHER_DB_TO_FACTOR = 10.0;

    /**
     * Hệ số chuyển đổi từ dB (attenuation) sang hệ số nhân ảnh hưởng tiêu thụ pin.
     * Giả định ảnh hưởng đến pin ít hơn ảnh hưởng đến trễ.
     */
    public static final double WEATHER_DRAIN_IMPACT_FACTOR = 20.0;

    /**
     * Hệ số chuyển đổi từ dB (attenuation) sang tỉ lệ loss tức thời do thời tiết.
     * Dùng trong {@code updateNodeStatus} để tính {@code weatherLoss}.
     */
    public static final double WEATHER_LOSS_FACTOR = 10.0;

    // ===========================================
    // 📊 GIỚI HẠN HỆ THỐNG (BOUNDS)
    // ===========================================

    /**
     * % pin tối thiểu. Node sẽ được coi là 'unhealthy' nếu pin dưới mức này.
     */
    public static final double MIN_BATTERY = 0.0; // Hoặc 10.0 nếu theo logic isHealthy()

    /**
     * % pin tối đa (dùng để kẹp giá trị nếu có sạc).
     */
    public static final double MAX_BATTERY = 100.0;

    /**
     * Tỉ lệ utilization tối đa (1.0 = 100%).
     * Dùng để kẹp giá trị của {@code resourceUtilization}.
     */
    public static final double MAX_UTILIZATION = 1.0;

    // ===========================================
    // 🌍 VẬT LÝ & MÔI TRƯỜNG
    // ===========================================

    /**
     * Hệ số chuyển đổi từ Mbps (Megabits/sec) sang Bps (Bytes/sec).
     * (1 Mbps = 1,000,000 bits/sec)
     * (1,000,000 bits/sec) / (8 bits/byte) = 125,000 Bytes/sec.
     */
    public static final double MBPS_TO_BPS_CONVERSION = 125000.0;
                
    /**
     * Thời gian chờ kết nối TCP mặc định (milliseconds).
     */
    public static final int TCP_CONNECT_TIMEOUT_MS = 1000; // 1 giây
}