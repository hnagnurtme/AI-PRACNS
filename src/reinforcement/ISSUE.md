# ISSUE: Kiểm tra & Tối ưu hóa RL, đảm bảo thuật toán RL vượt trội

**Vấn đề hiện tại:** Hệ thống chưa cập nhật neighbor động cho các node, dẫn đến các thuật toán baseline sử dụng thông tin neighbor cũ, gây ra kết quả sai lệch và đôi khi baseline lại tối ưu hơn RL. Cần khắc phục để đảm bảo baseline không vượt RL trong các kịch bản động.

**Lưu ý: Chỉ quan tâm và thực hiện các kiểm tra, tối ưu trong folder `reinforcement`. Không thực hiện hoặc đánh giá các file/folder ngoài `reinforcement`.**


## Mục tiêu
1. **Kiểm tra toàn bộ codebase** liên quan đến RL và các thuật toán baseline.
2. **Tối ưu cấu trúc code RL** (các file trong `agents/`, `environments/`, `training/`, `train_dynamic.py`, v.v.) để đảm bảo hiệu quả, dễ mở rộng, dễ bảo trì.
3. **Đảm bảo các thuật toán baseline (dijkstra, baseline, v.v.) luôn thua RL** trong các kịch bản hợp lý (so sánh kết quả trong `analysis/`).
4. **RL phải phát hiện được các node di chuyển**: Khi node di chuyển và không còn là neighbor, agent RL phải nhận biết và cập nhật lại neighbor list, tránh gửi gói tin sai.
5. **Tối ưu hiệu quả RL**: 
   - RL phải thích nghi tốt với môi trường động (node mobility, failures, v.v.).
   - Giảm thời gian hội tụ, tăng reward, giảm số lần gửi thất bại.
6. **Triển khai kiểm tra dữ liệu động**: Đảm bảo các dữ liệu đầu vào (topology, trạng thái node, traffic, failures, v.v.) thay đổi theo thời gian và RL thích nghi tốt với các thay đổi này.

8. **Triển khai mô phỏng cho thấy RL dự đoán trước node quá tải, mất cân bằng tài nguyên**: Thiết kế các kịch bản mà RL có thể phát hiện, dự đoán trước các node sắp quá tải hoặc mất cân bằng tài nguyên, từ đó chủ động phân phối lại lưu lượng/tài nguyên để tối ưu toàn mạng. Đánh giá khả năng RL tối ưu hơn baseline trong các trường hợp này, kể cả khi phải đánh đổi độ trễ để đạt cân bằng tài nguyên tốt hơn.

- [ ] Review toàn bộ code liên quan RL và các thuật toán baseline.
- [ ] Refactor/tối ưu các class RL agent, environment, replay buffer, policy, v.v.
- [ ] Đảm bảo RL agent cập nhật neighbor động khi node di chuyển (xem lại logic trong `network.py`, `node.py`, `dynamic_env.py`).
- [ ] Viết/kiểm tra lại các hàm so sánh kết quả RL với baseline (trong `analysis/`).
- [ ] Thêm/kiểm tra test case cho các trường hợp node di chuyển, neighbor thay đổi.
- [ ] Đảm bảo baseline luôn thua RL trong các báo cáo kết quả.
- [ ] Viết lại/tối ưu các hàm reward, state, action cho RL.
- [ ] Đánh giá lại hiệu quả RL qua các chỉ số: reward, success rate, convergence time.
- [ ] Triển khai kiểm tra với dữ liệu động: thay đổi topology, trạng thái node, traffic, failures trong quá trình huấn luyện và đánh giá.
- [ ] Mô phỏng và kiểm thử các mô hình truyền thông lỗi: packet loss, delay, node/link failure, v.v. Đánh giá khả năng phục hồi và thích nghi của RL.

## Checklist công việc
- [ ] Review toàn bộ code liên quan RL và các thuật toán baseline.
- [ ] Refactor/tối ưu các class RL agent, environment, replay buffer, policy, v.v.
- [ ] Đảm bảo RL agent cập nhật neighbor động khi node di chuyển (xem lại logic trong `network.py`, `node.py`, `dynamic_env.py`).
- [ ] Viết/kiểm tra lại các hàm so sánh kết quả RL với baseline (trong `analysis/`).
- [ ] Thêm/kiểm tra test case cho các trường hợp node di chuyển, neighbor thay đổi.
- [ ] Đảm bảo baseline luôn thua RL trong các báo cáo kết quả.
- [ ] Viết lại/tối ưu các hàm reward, state, action cho RL.
- [ ] Đánh giá lại hiệu quả RL qua các chỉ số: reward, success rate, convergence time.
- [ ] Triển khai kiểm tra với dữ liệu động: thay đổi topology, trạng thái node, traffic, failures trong quá trình huấn luyện và đánh giá.
- [ ] Mô phỏng và kiểm thử các mô hình truyền thông lỗi: packet loss, delay, node/link failure, v.v. Đánh giá khả năng phục hồi và thích nghi của RL.
- [ ] Thiết kế và thực hiện test đảm bảo hoàn thiện hệ thống, bao gồm tối ưu phân bổ tài nguyên mạng, giảm tỷ lệ mất gói (packet loss), tối ưu cả độ trễ lẫn các chỉ số chất lượng dịch vụ khác (không chỉ riêng độ trễ).
- [ ] Thiết kế mô phỏng cho thấy RL dự đoán trước node quá tải, mất cân bằng tài nguyên và chủ động tối ưu phân phối tài nguyên/lưu lượng, kể cả khi phải đánh đổi độ trễ để đạt hiệu quả tổng thể tốt hơn baseline.

- Sử dụng cấu trúc OOP rõ ràng cho agent, environment.
- Tách biệt rõ logic RL và baseline.
- Sử dụng logging, checkpoint, tensorboard để theo dõi quá trình huấn luyện.
- Tối ưu lại hàm phát hiện neighbor động (có thể dùng graph hoặc spatial index).
- Đảm bảo code dễ mở rộng cho các thuật toán RL khác (A2C, PPO, v.v.).
- Tạo các kịch bản kiểm thử tự động cho dữ liệu động và mô hình truyền thông lỗi.
## Gợi ý cải tiến
- Sử dụng cấu trúc OOP rõ ràng cho agent, environment.
- Tách biệt rõ logic RL và baseline.
- Sử dụng logging, checkpoint, tensorboard để theo dõi quá trình huấn luyện.
- Tối ưu lại hàm phát hiện neighbor động (có thể dùng graph hoặc spatial index).
- Đảm bảo code dễ mở rộng cho các thuật toán RL khác (A2C, PPO, v.v.).
- Tạo các kịch bản kiểm thử tự động cho dữ liệu động và mô hình truyền thông lỗi.
- Đánh giá và tối ưu đồng thời nhiều chỉ số: phân bổ tài nguyên, tỷ lệ mất gói, throughput, fairness, không chỉ riêng độ trễ.
