# So sánh MILP vs ALNS+Tabu cho bài toán CVRPTW

## 1. Tổng quan bài toán

Cả hai phương pháp cùng giải **CVRPTW** (Capacitated Vehicle Routing Problem with Time Windows) với:
- **20 cửa hàng** + 1 depot
- **Hàm mục tiêu**: minimize `Transport + Late penalty (λ₁) + Congestion (λ₂)`
- **Ràng buộc**: capacity 1800kg, time-window (soft upper → penalty), congestion

---

## 2. Bảng so sánh kết quả chi tiết

### MILP (CBC Solver, time limit = 120s/mỗi K)

| K | Active | Transport | Late Penalty | Congestion | **TOTAL COST** | Time |
|:-:|:------:|----------:|-------------:|-----------:|---------------:|-----:|
| 3 | 3 | 1,466,640 | 158,000 | 160,000 | 1,784,640 | 117s |
| 4 | 4 | 1,891,260 | 9,734,000 | 175,000 | **11,800,260** | 117s |
| 5 | 3 | 1,597,140 | 3,534,000 | 160,000 | 5,291,140 | 117s |
| 6 | 3 | 1,481,940 | 0 | 150,000 | 1,631,940 | 120s |
| 7 | 3 | 1,650,060 | 0 | 160,000 | 1,810,060 | 120s |
| 8 | 4 | 2,123,640 | 3,902,000 | 180,000 | **6,205,640** | 120s |
| 9 | 3 | 1,459,440 | 0 | 155,000 | **1,614,440 ★** | 120s |
| 10 | 3 | 1,550,880 | 0 | 160,000 | 1,710,880 | 120s |
| 11 | 3 | 1,427,940 | 5,574,000 | 155,000 | 7,156,940 | 120s |
| 12 | 3 | 1,478,700 | 0 | 150,000 | 1,628,700 | 120s |
| 13 | 4 | 1,830,600 | 60,000 | 150,000 | 2,040,600 | 120s |

### ALNS + Tabu Search (5000 iterations/mỗi K)

| K | Active | Transport | Late Penalty | Congestion | **TOTAL COST** | Time |
|:-:|:------:|----------:|-------------:|-----------:|---------------:|-----:|
| 3 | 3 | 1,442,880 | 0 | 150,000 | **1,592,880 ★** | 15.0s |
| 4 | 3 | 1,442,880 | 0 | 150,000 | **1,592,880 ★** | 15.0s |
| 5 | 3 | 1,442,880 | 0 | 150,000 | **1,592,880 ★** | 15.1s |
| 6 | 3 | 1,442,880 | 0 | 150,000 | **1,592,880 ★** | 15.2s |
| 7 | 3 | 1,442,880 | 0 | 150,000 | **1,592,880 ★** | 15.8s |
| 8 | 3 | 1,442,880 | 0 | 150,000 | **1,592,880 ★** | 15.8s |
| 9 | 3 | 1,442,880 | 0 | 150,000 | **1,592,880 ★** | 15.9s |
| 10 | 3 | 1,442,880 | 0 | 150,000 | **1,592,880 ★** | 16.3s |
| 11 | 3 | 1,442,880 | 0 | 150,000 | **1,592,880 ★** | 16.5s |
| 12 | 3 | 1,442,880 | 0 | 150,000 | **1,592,880 ★** | 16.4s |
| 13 | 3 | 1,442,880 | 0 | 150,000 | **1,592,880 ★** | 16.1s |

---

## 3. So sánh trực tiếp (Best vs Best)

| Tiêu chí | MILP (K=9) | ALNS+Tabu | Chênh lệch |
|---|---:|---:|---:|
| **Tổng chi phí** | 1,614,440 | **1,592,880** | **-21,560 (-1.3%)** |
| Transport | 1,459,440 | **1,442,880** | -16,560 |
| Late penalty | 0 | 0 | 0 |
| Congestion | 155,000 | **150,000** | -5,000 |
| Xe active | 3 | 3 | 0 |
| Thời gian/K | 120s | **~15s** | **~8x nhanh hơn** |
| Tổng thời gian (11 K) | ~22 phút | **~3 phút** | **~7x nhanh hơn** |

---

## 4. Phân tích chi tiết

### 4.1 MILP — Điểm mạnh & Hạn chế

**Điểm mạnh:**
- Mô hình toán học chính xác, đảm bảo tính tối ưu **nếu có đủ thời gian**
- Dễ mở rộng thêm ràng buộc mới (chỉ cần thêm bất đẳng thức)

**Hạn chế quan trọng:**

> [!WARNING]
> Kết quả MILP **không ổn định** giữa các giá trị K. Chi phí dao động từ **1.6 triệu đến 11.8 triệu VND** — chênh lệch lên tới **7x**. Nguyên nhân: solver CBC bị giới hạn 120 giây, không đủ thời gian tìm lời giải tối ưu cho mọi cấu hình K.

- K=4: 11.8 triệu (late penalty = 9.7 triệu!) — solver trả về lời giải kém
- K=8: 6.2 triệu (late penalty = 3.9 triệu!) — tương tự
- K=11: 7.1 triệu (late penalty = 5.5 triệu!) — tương tự
- Nghĩa là: khi K tăng, số biến quyết định tăng theo O(n² × K), bài toán trở nên khó hơn cho solver

### 4.2 ALNS+Tabu — Điểm mạnh & Hạn chế

**Điểm mạnh:**
- **Ổn định 100%**: Cùng kết quả 1,592,880 VND cho mọi K từ 3→13
- **Nhanh hơn ~8x**: ~15s vs 120s mỗi K
- **Late penalty = 0**: Luôn giao hàng đúng giờ
- Tự động phát hiện chỉ cần 3 xe là tối ưu, bất kể K cho phép bao nhiêu
- Các destroy/repair operators đa dạng giúp khám phá không gian lời giải hiệu quả

**Hạn chế:**
- Là meta-heuristic → **không đảm bảo tìm được lời giải tối ưu toàn cục**
- Kết quả có thể thay đổi nếu đổi random seed
- Chất lượng phụ thuộc vào tuning tham số (iterations, cooling rate, tenure...)

### 4.3 Tại sao ALNS+Tabu lại tốt hơn MILP ở đây?

> [!IMPORTANT]
> ALNS+Tabu **không thực sự tốt hơn** MILP về mặt lý thuyết. Vấn đề nằm ở **time limit 120s** của CBC solver.

MILP với solver CBC (open-source) gặp khó với bài toán 20 customers vì:
1. **Số biến binary**: `x[i,j,k]` = n² × K = 441 × K biến 0/1
2. **Relaxation gap**: LP relaxation yếu → branch-and-bound chậm hội tụ
3. **Big-M constraints**: Ràng buộc thời gian dùng big-M → relaxation lỏng lẻo

Nếu tăng time limit lên 10-30 phút hoặc dùng solver thương mại (Gurobi, CPLEX), MILP có thể cho kết quả tốt hơn hoặc bằng ALNS+Tabu.

---

## 5. Đánh giá: Có đáp ứng yêu cầu tối ưu hóa chi phí không?

### ✅ CẢ HAI đều đáp ứng yêu cầu tối ưu hóa chi phí

| Yêu cầu | MILP | ALNS+Tabu |
|---|:---:|:---:|
| Tối thiểu hóa chi phí vận chuyển | ✅ | ✅ |
| Xét đến congestion (tắc đường) | ✅ | ✅ |
| Xét đến time window (giờ giao hàng) | ✅ | ✅ |
| Phạt trễ hẹn (late penalty) | ✅ | ✅ |
| Tối ưu số xe sử dụng | ✅ | ✅ |
| Phục vụ tất cả cửa hàng | ✅ | ✅ |
| Kết quả ổn định | ⚠️ Không ổn định | ✅ Rất ổn định |
| Thời gian chạy hợp lý | ⚠️ Chậm (~2 phút/K) | ✅ Nhanh (~15s/K) |

### Kết luận

> [!TIP]
> **Cho bài toán thực tế với 20 cửa hàng này:**
> - **ALNS+Tabu là lựa chọn tốt hơn** cho sản xuất: nhanh, ổn định, kết quả tốt
> - **MILP phù hợp cho nghiên cứu/benchmark**: đảm bảo tối ưu toàn cục nếu có đủ thời gian
> - Khi quy mô bài toán tăng (50-100+ customers), MILP sẽ càng khó giải trong thời gian hợp lý, ALNS+Tabu vẫn scale tốt

Cả hai phương pháp đều **đáp ứng yêu cầu tối ưu hóa chi phí đầu ra**, nhưng ALNS+Tabu cho kết quả **thực tế tốt hơn** trong điều kiện tài nguyên tính toán giới hạn.

---

## 6. Lời giải tối ưu: 3 xe, 1,592,880 VND

```
Vehicle 1:  Depot → MC-182 → BB-8 → BB-97 → BB-252 → MC-38 → BB-31 → MC-111 → MC-161 → Depot
            (8 stores, 850.7 kg, 28.38 km)

Vehicle 2:  Depot → BB-351 → BB-847 → Depot
            (2 stores, 275.6 kg, 8.84 km)

Vehicle 3:  Depot → MC-8 → BB-6 → BB-388A → MC-128 → MC-5A → BB-279 → MC-41 → BB-112 → MC-122 → MC-390 → Depot
            (10 stores, 1118.8 kg, 42.94 km)
```

> Tổng: 20/20 cửa hàng | 0 phút trễ | 80.16 km | 1,592,880 VND
