# VRP Optimize Solver: T-ALNS-RRD

Dự án này chứa mã nguồn để giải quyết bài toán Điều phối Xe cộ (Vehicle Routing Problem - VRP) với cửa sổ thời gian (Time Windows) và ưu tiên thời gian thực. Cụ thể, trong dự án cung cấp thuật toán **T-ALNS-RRD** (Tabu-guided Adaptive Large Neighborhood Search combined with Rollout Reassignment & Dispatch).

## Yêu cầu hệ thống

- Python 3.8 trở lên.
- Quản lý gói thư viện `pip`.

## Hướng dẫn cài đặt và chạy

**Bước 1: Clone kho lưu trữ về máy phân tích**

Mở terminal/command prompt và thực thi lệnh sau:
```bash
git clone git@github.com:rudojr/VRP_Optimize.git
cd VRP_Optimize
```

*(Lưu ý: Thay thế URL nếu thư mục của bạn có URL git khác)*

**Bước 2: Cài đặt các thư viện cần thiết**

Nên tạo một virtual environment (như bạn đang có là thư mục `venv/`) trước khi cài các module:

```bash
# Tạo môi trường ảo (tuỳ chọn)
python -m venv venv
source venv/bin/activate  # (Với MacOS/Linux) hoặc venv\Scripts\activate trên Windows

# Cài đặt thư viện
pip install numpy pandas
```

**Bước 3: Chạy Script giải lập T-ALNS-RRD**

Tệp nguồn chính nằm ở thư mục `src/`, và tệp `vrp_alns_tabu_rrd.py` đã được cấu hình chạy tự động nạp các tệp điểm dữ liệu từ thư mục `data/` với "fleet sweep" quét để tự tính với số xe trong tập tin.

Chạy lệnh sau ngay từ thư mục gốc của dự án:
```bash
python src/vrp_alns_tabu_rrd.py
```

## Cách thức hoạt động

Khi chạy tệp `vrp_alns_tabu_rrd.py`:
- Script sẽ tự động lấy thư mục gốc của project làm thư mục thực thi hiện tại (nhờ lệnh `sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))`).
- Nó sẽ đọc các tệp dữ liệu về cửa hàng, và hai ma trận thời gian & khoảng cách từ mục `data/`:
  - `data/140_stores.csv`
  - `data/140_distant_matrix.csv`
  - `data/140_time_matrix.csv`
- Script bắt đầu chạy tiến trình **Fleet Sweep**: thực hiện quét nghiệm VRP với số lượng xe (K) từ 3 đến 13 xe liên tục để tìm ra kết quả tối ưu.
- Quá trình sẽ in log chạy trực tiếp trên console cho từng số lượng xe (K) để bạn có thể xem cập nhật về **Chi phí (VND)**, **Tỷ lệ đúng hạn (OTDR)**, và **Độ tắc nghẽn (CES)**.
- Cuối cùng, một **bảng tổng hợp** (RESULTS — Tabu-ALNS-Rollout) sẽ được hiển thị kèm theo dấu `<star>` cho kết quả hiệu suất tốt nhất.

## Một số tuỳ chọn bổ sung (dành cho Nhà phát triển)

Nếu bạn muốn tùy chỉnh cấu hình chi tiết (như số vòng lặp, thay đổi số xe cố định, bật tắt tùy chọn in log quá trình, v.v.), bạn có thể truy cập thẳng vào trong tệp `src/vrp_alns_tabu_rrd.py` (khối `if __name__ == "__main__":` dòng 934) và thay đổi trực tiếp biến gọi hàm `rrd_solve(...)`:

```python
        result = rrd_solve(
            data,
            num_vehicles=K,         # Số lượng xe
            alns_iterations=500,    # Số vòng lặp Phase 1 (ALNS)
            reassign_iter=10,       # Số vòng lặp Phase 2a 
            verbose=False,          # Tắt bật in báo cáo log chuyên sâu
        )
```
