# Plan: Parallel Seam Carving bằng OpenMP

Bài tập nhóm cuối kì môn Tính toán song song. Nhóm 3 người, thời gian 3 tuần.

**Ngày bắt đầu (N1):** `__/__/____` · **Hạn nộp (N21):** `__/__/____`

## Cách dùng file này

1. Tìm tên role của mình (A, B hoặc C) ở [mục 5](#5-phân-công).
2. Mở [mục 7](#7-bảng-công-việc-chi-tiết), tìm tuần hiện tại, làm task đầu tiên chưa tick của mình.
3. Mỗi task ghi rõ: làm trong file nào, cần task nào xong trước, và "xong khi" nào.
4. Làm xong thì đổi `- [ ]` thành `- [x]` ngay trong file này và commit cùng với code.
5. Bị chặn quá nửa ngày thì báo nhóm, đừng tự kẹt.

Kí hiệu: `N3` là ngày thứ 3 tính từ ngày bắt đầu. `A2` là task số 2 của role A.

---

## 1. Mục tiêu

Cài đặt song song thuật toán **Seam Carving** (Avidan & Shamir, SIGGRAPH 2007) để co ảnh theo nội dung, dùng **OpenMP**.

Môn học chấm phần *song song hoá*, không chấm xử lý ảnh. Bốn thứ phải có:

1. Bản tuần tự đúng, làm baseline.
2. Phân tích phụ thuộc dữ liệu: chỗ nào song song được, vì sao.
3. Các phiên bản OpenMP từ ngây thơ đến tối ưu.
4. Đo đạc nghiêm túc (speedup, efficiency, Amdahl, Karp–Flatt) và giải thích được kết quả.

**Kỳ vọng thực tế:** tài liệu tìm được cho thấy OpenMP trên seam carving chuẩn chỉ cho speedup khiêm tốn, có bài báo cáo gần như không tăng tốc ([mục 10](#10-tài-liệu-tham-khảo)). Lý do: bước quy hoạch động phụ thuộc theo hàng và bài toán bị giới hạn bởi băng thông bộ nhớ. Điểm mạnh của bài phải nằm ở phần phân tích, không phải ở con số speedup.

**Công cụ thống nhất cả nhóm:**

| Hạng mục | Lựa chọn |
|---|---|
| Ngôn ngữ | C++17 + OpenMP |
| Compiler | GCC qua MSYS2/MinGW-w64 (hoặc WSL). **Không dùng MSVC**: `/openmp` mặc định chỉ hỗ trợ OpenMP 2.0 |
| Build | CMake |
| Đọc/ghi ảnh | `stb_image.h`, `stb_image_write.h` |
| Vẽ biểu đồ | Python + matplotlib |
| Sản phẩm nộp | Mã nguồn, báo cáo, slide, demo |

---

## 2. Thuật toán và phân tích song song

Xoá 1 seam dọc trên ảnh H×W gồm 4 bước, lặp lại k lần để giảm k cột:

| Bước | Công việc | Độ phức tạp | Phụ thuộc dữ liệu | Khả năng song song |
|---|---|---|---|---|
| 1. Energy | `e(i,j) = \|∂I/∂x\| + \|∂I/∂y\|` | O(HW) | Không có | Rất dễ: `parallel for` |
| 2. Cumulative energy (DP) | `M(i,j) = e(i,j) + min(M(i-1,j-1), M(i-1,j), M(i-1,j+1))` | O(HW) | Hàng i phụ thuộc hàng i-1 | Khó: chỉ song song trong một hàng, hoặc phải phân rã miền |
| 3. Tìm seam | argmin hàng cuối, backtrack từ dưới lên | O(W + H) | Backtrack tuần tự | argmin song song được; backtrack để tuần tự |
| 4. Xoá seam | Mỗi hàng dịch trái 1 pixel từ vị trí seam | O(HW) | Không có | Dễ: `parallel for` theo hàng |

**Điểm khó cốt lõi (phần "ăn điểm" của báo cáo):**

- Vòng lặp ngoài (k seam) **không** song song được: seam sau phụ thuộc ảnh sau khi xoá seam trước.
- DP song song trong một hàng cần một barrier sau mỗi hàng. Mỗi hàng chỉ có W phần tử nên overhead đồng bộ có thể nuốt hết lợi ích.
- Bài toán memory-bound, speedup bão hoà sớm.

---

## 3. Các phiên bản cài đặt

| Phiên bản | Mức | Nội dung | Mục đích |
|---|---|---|---|
| **V0 – Sequential** | Bắt buộc | Tuần tự, tối ưu hợp lý (mảng 1D liên tục, `-O2`) | Baseline tính speedup |
| **V1 – Naive OpenMP** | Bắt buộc | `#pragma omp parallel for` cho energy, từng hàng DP, xoá seam; mỗi hàng DP mở một parallel region | Cho thấy overhead fork/join |
| **V2 – Tuned OpenMP** | Bắt buộc, bản chính | Một `parallel` region bao toàn bộ DP, `omp for` từng hàng; `schedule(static)`; argmin song song; `simd` cho energy | Bản nộp tối thiểu |
| **V3 – Triangle decomposition** | Mục tiêu, hạn chót N17 | Chia DP thành các dải nhiều hàng, mỗi dải chạy 2 vòng song song (chi tiết ở dưới) | Tăng độ hạt, giảm số lần đồng bộ |
| **V4 – Incremental energy** | Tuỳ chọn | Sau khi xoá seam chỉ tính lại energy quanh seam | Thay đổi tỉ lệ các pha |

**Yêu cầu tính đúng:** V1–V4 phải cho ảnh **giống từng bit** với V0. Quy tắc phá hoà: khi bằng nhau luôn chọn **cột trái nhất**, ở cả bước `min` trong DP, bước argmin hàng cuối, và bước backtrack.

**Đã bỏ khỏi phạm vi:** forward energy, seam ngang, xoá nhiều seam mỗi vòng, phóng to ảnh. Chỉ nhắc ở "hướng phát triển".

### Cách V3 hoạt động

Giá trị `M(i,j)` phụ thuộc vào một tam giác ngược phía trên nó. Tận dụng điều đó:

- Chia các hàng thành **dải** cao `s` hàng. Trong mỗi dải, chia các cột thành khối rộng `b = 2s`.
- **Vòng 1 (tam giác xuôi, các khối chạy song song):** ở hàng thứ `r` của dải (r = 0…s−1), khối `k` tính các cột `[k·b + r, (k+1)·b − r)`. Mỗi hàng hẹp dần 1 cột mỗi bên, nên chỉ cần dữ liệu của chính khối đó ở hàng trên.
- **Vòng 2 (tam giác ngược, chạy song song):** lấp phần còn trống quanh mỗi ranh giới khối. Ở hàng `r` (r = 1…s−1), ranh giới `k` tính các cột `[k·b − r, k·b + r)`, cắt theo biên ảnh `[0, W)`.
- Xong một dải thì sang dải kế. Hàng đầu của dải mới dùng hàng cuối (đã đầy đủ) của dải trước.

Kết quả: mỗi dải chỉ cần 2 lần đồng bộ thay vì `s` lần. Tài liệu gợi ý `b` khoảng 64–90 pixel. Nguồn: bài blog của Sam Westrick ở mục 10.

---

## 4. Cấu trúc project

```
parallel-seam-carving-openmp/
├── PLAN.md
├── README.md                 # cách build, chạy, tái lập thí nghiệm      (A)
├── CMakeLists.txt            #                                           (C)
├── third_party/              # stb_image.h, stb_image_write.h            (A)
├── include/
│   ├── image.hpp             # struct Image, load/save                   (A)
│   ├── seam_carving.hpp      # interface chung                           (A)
│   └── timer.hpp             # đo thời gian theo pha                     (C)
├── src/
│   ├── image.cpp             #                                           (A)
│   ├── seq.cpp               # V0                                        (A)
│   ├── omp_naive.cpp         # V1: DP (B), energy + remove (C)
│   ├── omp_tuned.cpp         # V2: DP (B), argmin (A), energy + remove (C)
│   ├── omp_triangle.cpp      # V3: DP                                    (B)
│   └── main.cpp              # CLI                                       (A)
├── tests/
│   ├── test_seq.cpp          # V0 trên ma trận tính tay                  (A)
│   ├── test_correctness.cpp  # so từng bit với V0                        (A)
│   └── test_dp.cpp           # test riêng bước DP cho V3                 (A)
├── data/                     # ảnh test nhỏ                              (C)
├── scripts/
│   ├── make_test_images.py   # sinh ảnh các kích thước                   (C)
│   ├── run_benchmark.ps1     # chạy ma trận thí nghiệm, ghi CSV          (C)
│   └── plot.py               # vẽ biểu đồ                                (C)
├── results/                  # CSV + hình (không commit file sinh ra)
└── report/
    ├── notes/                # ghi chú thiết kế, hình vẽ nháp
    └── ...                   # báo cáo + slide
```

### Interface đề xuất (A chốt ở task A2)

```cpp
// image.hpp
struct Image {
    int width;      // chiều rộng hiện tại, giảm dần sau mỗi seam
    int height;
    int stride;     // chiều rộng ban đầu; buffer không cấp phát lại
    std::vector<uint8_t> data;   // RGB, kích thước height * stride * 3
};

// seam_carving.hpp — mỗi phiên bản là một namespace: seq, naive, tuned, triangle
void compute_energy(const Image& img, std::vector<int32_t>& E);
void compute_cumulative(const std::vector<int32_t>& E, int w, int h, int stride,
                        std::vector<int32_t>& M);
void find_seam(const std::vector<int32_t>& M, int w, int h, int stride,
               std::vector<int>& seam);          // seam[i] = cột bị xoá ở hàng i
void remove_seam(Image& img, const std::vector<int>& seam);
```

Các quyết định đi kèm:

- **Energy và M dùng số nguyên `int32_t`**, không dùng float. So sánh từng bit sẽ đơn giản và không lo sai số làm tròn.
- **Energy:** tổng trên 3 kênh RGB của `|trái − phải| + |trên − dưới|`; ở biên thì lặp lại pixel biên.
- **Buffer giữ nguyên `stride`:** xoá seam chỉ là `memmove` trong từng hàng rồi giảm `width`. Không cấp phát lại mỗi vòng.
- `main.cpp` chọn phiên bản qua một bảng con trỏ hàm, nên các phiên bản có thể dùng lẫn hàm của nhau (ví dụ V3 dùng energy và remove của V2).

### CLI

```
seamcarve --input a.jpg --output b.png --remove-cols 300 --version tuned --threads 8 --timing
```

- `--version`: `seq` | `naive` | `tuned` | `triangle`
- `--timing`: in một dòng CSV `version,threads,width,height,seams,t_energy,t_dp,t_find,t_remove,t_total` (giây). Không tính thời gian đọc/ghi ảnh.
- `--save-seams out.png` (tuỳ chọn): lưu ảnh gốc có vẽ các seam, dùng cho demo.

---

## 5. Phân công

| Role | Người | Việc chính | Tỉ lệ ước lượng |
|---|---|---|---|
| **A – Baseline & kiểm thử** | `________` | `image`, V0, CLI, toàn bộ test, argmin song song của V2, test và gỡ lỗi V3 cùng B, README, slide | ~33% |
| **B – Song song hoá DP** | `________` | DP của V1, V2, V3; ghi chú phân tích phụ thuộc | ~34% |
| **C – Hiệu năng & thí nghiệm** | `________` | CMake, timer, energy và remove song song, script đo, biểu đồ, ghép báo cáo | ~33% |

Phần báo cáo của mỗi người:

| Chương | Người viết |
|---|---|
| 1. Giới thiệu | C |
| 2. Cơ sở lý thuyết | A |
| 3. Cài đặt tuần tự, breakdown thời gian | A |
| 4. Phân tích khả năng song song | B |
| 5. Thiết kế song song V1 → V2 → V3 | B (C viết đoạn energy/remove, A viết đoạn argmin) |
| 6. Kiểm chứng tính đúng | A |
| 7. Thực nghiệm | C |
| 8. Thảo luận | C, B góp ý |
| 9. Kết luận | C |

Mỗi người đều tự tay viết ít nhất một đoạn code OpenMP và phải hiểu toàn bộ code để trả lời vấn đáp.

---

## 6. Quy trình làm việc

**Git:**

```bash
git clone https://github.com/duongdotung09-cell/parallel-seam-carving-openmp.git
git checkout -b a/v0-seq          # tên nhánh: <role>/<việc>
# ... code ...
git add <file>; git commit -m "A3: V0 compute_energy"
git pull --rebase origin main     # cập nhật trước khi push
git push -u origin a/v0-seq       # rồi mở pull request trên GitHub
```

- Bắt đầu message commit bằng mã task (`A3: ...`) để dễ tra.
- Một người khác xem pull request rồi mới merge. Pull request nên nhỏ, mỗi task một cái.
- `main` phải luôn build được và qua test.

**Định nghĩa "xong" chung cho mọi task code:**

1. Build không cảnh báo với `-Wall -Wextra`.
2. Test hiện có vẫn qua.
3. Đã merge vào `main`.
4. Đã tick trong file này.

**Họp:** 15 phút vào N1, N3, N5, N7, N10, N12, N14, N17, N19. Mỗi người nói: đã xong gì, đang làm gì, bị chặn bởi gì.

---

## 7. Bảng công việc chi tiết

### Tuần 1 (N1–N7): baseline và bản ngây thơ

**Cả nhóm**

- [ ] **S1 – Cài môi trường** (N1)
  Cài MSYS2 + `mingw-w64-x86_64-gcc`, CMake, Git, Python + matplotlib. Clone repo.
  Xong khi: biên dịch và chạy được chương trình OpenMP in ra số thread (`g++ -fopenmp`).
- [ ] **S2 – Đọc paper gốc** (N1–N2)
  Đọc mục 3 (energy, seam, quy hoạch động) và xem các hình minh hoạ.
  Xong khi: mỗi người tự giải thích được 4 bước ở mục 2 của file này.

**Role A**

- [ ] **A1 – `Image` và đọc/ghi ảnh** (N1–N2) · file: `third_party/`, `include/image.hpp`, `src/image.cpp` · cần: S1
  Tải 2 header stb; viết `load_image`, `save_image` (PNG).
  Xong khi: đọc một ảnh JPG rồi ghi ra PNG, mở lên giống ảnh gốc.
- [ ] **A2 – Chốt interface** (N3) · file: `include/seam_carving.hpp` · cần: A1
  Dựa trên đề xuất ở mục 4; báo cả nhóm khi merge. **B2 và C4 đang chờ task này.**
  Xong khi: header đã merge, B và C xác nhận đã đọc.
- [ ] **A3 – V0 tuần tự** (N3–N5) · file: `src/seq.cpp` · cần: A2
  Viết 4 hàm trong namespace `seq`. Chú ý quy tắc phá hoà chọn cột trái nhất.
  Xong khi: xoá 100 seam trên một ảnh, kết quả nhìn hợp lý (vật thể chính không bị méo).
- [ ] **A4 – CLI** (N5–N6) · file: `src/main.cpp` · cần: A3, C2
  Đọc tham số, vòng lặp xoá k seam, gọi timer theo pha, in dòng CSV khi có `--timing`.
  Xong khi: lệnh mẫu ở mục 4 chạy được với `--version seq`.
- [ ] **A5 – Test V0 và khung so từng bit** (N6–N7) · file: `tests/test_seq.cpp`, `tests/test_correctness.cpp` · cần: A3
  Test V0 trên ma trận energy 4×5 tính tay (so M và seam). Viết hàm so hai ảnh từng byte.
  Xong khi: `ctest` chạy và qua; so được V1 với V0 trên 1 ảnh.

**Role B**

- [ ] **B1 – Phân tích phụ thuộc của DP** (N1–N3) · file: `report/notes/dp-dependency.md` · cần: S2
  Vẽ đồ thị phụ thuộc của `M(i,j)`; đọc bài blog của Westrick; ghi lại vì sao hàng i không thể tính trước hàng i−1 và vì sao các ô trong một hàng độc lập.
  Xong khi: có 1 hình vẽ và nửa trang ghi chú, dùng lại được cho chương 4.
- [ ] **B2 – DP của V1** (N4–N5) · file: `src/omp_naive.cpp` · cần: A2
  `compute_cumulative` với `#pragma omp parallel for` đặt bên trong vòng lặp hàng (mỗi hàng một parallel region).
  Xong khi: M giống hệt M của V0 trên 1 ảnh, thử với 1, 2, 4 thread.
- [ ] **B3 – Ghi chú thiết kế V2 và V3** (N6–N7) · file: `report/notes/dp-design.md` · cần: B1
  V2: mã giả với một parallel region. V3: chọn `s` và `b`, viết công thức khoảng cột của 2 vòng, liệt kê trường hợp biên (W không chia hết cho b, dải cuối thấp hơn `s`).
  Xong khi: A đọc và hiểu đủ để viết test cho V3 (task A8).

**Role C**

- [ ] **C1 – Khung CMake** (N1–N2) · file: `CMakeLists.txt` · cần: S1
  Target `seamcarve` và các target test; `find_package(OpenMP REQUIRED)`; cờ `-O2 -Wall -Wextra`; bật `ctest`.
  Xong khi: `cmake -B build -G "MinGW Makefiles"` và `cmake --build build` chạy được với một `main.cpp` rỗng.
- [ ] **C2 – Timer** (N2–N3) · file: `include/timer.hpp` · cần: C1
  Lớp cộng dồn thời gian theo tên pha, dùng `omp_get_wtime()`. **A4 đang chờ task này.**
  Xong khi: có ví dụ dùng trong comment đầu file; đo được 2 pha giả và in ra tổng.
- [ ] **C3 – Ảnh test** (N3–N4) · file: `data/`, `scripts/make_test_images.py`
  Chọn 2–3 ảnh gốc có bản quyền tự do (phong cảnh có vật thể rõ). Script phóng ra 3 kích thước: 1280×720, 1920×1080, 3840×2160. Chỉ commit ảnh gốc nhỏ.
  Xong khi: chạy script ra đủ 3 kích thước cho mỗi ảnh.
- [ ] **C4 – Energy và remove của V1** (N4–N6) · file: `src/omp_naive.cpp` · cần: A2
  `compute_energy` và `remove_seam` với `#pragma omp parallel for` theo hàng.
  Xong khi: kết quả giống hệt V0 trên 1 ảnh, thử với 1, 2, 4 thread.
- [ ] **C5 – Script benchmark** (N6–N7) · file: `scripts/run_benchmark.ps1` · cần: A4
  Lặp qua version × threads × ảnh, mỗi cấu hình 5 lần, gom các dòng CSV vào `results/raw.csv`.
  Xong khi: chạy được với `seq` và `naive` trên 1 ảnh, ra file CSV đúng cột.

**Mốc M1 (cuối N7):** V0 và V1 chạy đúng. Có breakdown thời gian từng pha của V0 trên ảnh 1080p. C gửi con số này cho cả nhóm vì nó là đầu vào cho phân tích Amdahl.

### Tuần 2 (N8–N14): bản chính V2 và bắt đầu V3

**Role A**

- [ ] **A6 – Ma trận test đầy đủ** (N8–N9) · file: `tests/test_correctness.cpp` · cần: A5, B2, C4
  So từng bit với V0: mọi phiên bản × 3 ảnh × số thread 1, 2, 3, 4, 8 × lặp 10 lần. Thêm ảnh biên: rất hẹp (W = 3), một màu (mọi energy bằng nhau, kiểm tra phá hoà).
  Xong khi: `ctest` qua với V1; thêm phiên bản mới chỉ cần thêm 1 dòng.
- [ ] **A7 – Argmin song song cho V2** (N9–N10) · file: `src/omp_tuned.cpp` (hàm `find_seam`) · cần: A6
  Tìm min hàng cuối song song: mỗi thread giữ min cục bộ kèm chỉ số, gộp lại sao cho khi bằng nhau lấy chỉ số nhỏ nhất. Backtrack giữ tuần tự.
  Xong khi: seam giống hệt V0 trên ảnh một màu và ảnh thường, với mọi số thread.
- [ ] **A8 – Test riêng cho bước DP** (N11–N12) · file: `tests/test_dp.cpp` · cần: B3
  Sinh energy ngẫu nhiên (seed cố định), so M của một phiên bản với M của V0. Quét W từ 1 đến 300 và H từ 1 đến 100 để bắt lỗi biên của tam giác.
  Xong khi: test qua với V2; sẵn sàng cho B chạy với V3.
- [ ] **A9 – Kiểm tra data race** (N13) · cần: B4, C6
  Build trên WSL với `-fsanitize=thread -g`, chạy V1 và V2 với 4 thread.
  Xong khi: không có cảnh báo, hoặc đã báo lỗi cho người viết đoạn đó.
- [ ] **A10 – Nháp chương 2 và 3** (N13–N14) · file: `report/`
  Chương 2: seam carving, quy hoạch động, OpenMP. Chương 3: thiết kế V0, độ phức tạp, breakdown từ M1.
  Xong khi: có bản nháp đầy đủ ý, chưa cần trau chuốt.

**Role B**

- [ ] **B4 – DP của V2** (N8–N10) · file: `src/omp_tuned.cpp` · cần: B2
  Một `#pragma omp parallel` bao vòng lặp hàng, bên trong `#pragma omp for schedule(static)`. So thời gian DP với V1 và V0.
  Xong khi: `test_correctness` qua; có số đo thời gian DP của V0/V1/V2 trên ảnh 1080p với 1, 2, 4, 8 thread.
- [ ] **B5 – DP của V3** (N11–N14) · file: `src/omp_triangle.cpp` · cần: B3, B4, A8
  Cài 2 vòng tam giác theo mô tả ở mục 3. Làm đúng trước với một thread, sau đó mới thêm `omp for`. Chạy `test_dp` liên tục.
  Xong khi: `test_dp` qua với 1 thread. (Đúng với nhiều thread là mục tiêu của B7.)
- [ ] **B6 – Nháp chương 4** (N14) · file: `report/` · cần: B1
  Dựa trên `dp-dependency.md`: đồ thị phụ thuộc, phần nào song song được, dự đoán Amdahl từ breakdown M1.
  Xong khi: có bản nháp đầy đủ ý.

**Role C**

- [ ] **C6 – Energy và remove của V2** (N8–N10) · file: `src/omp_tuned.cpp` · cần: C4
  Energy: `parallel for` theo hàng, `simd` cho vòng trong. Remove: `memmove` từng hàng, `parallel for` theo hàng. Thử `static` / `dynamic` / `guided` và ghi lại số đo.
  Xong khi: `test_correctness` qua; có bảng so sánh 3 kiểu schedule.
- [ ] **C7 – Benchmark sơ bộ** (N11) · cần: B4, C6, C5
  Chạy V0/V1/V2 trên ảnh 1080p với 1, 2, 4, 8 thread.
  Xong khi: gửi nhóm bảng speedup sơ bộ; nếu V2 không nhanh hơn V0 thì báo ngay để B và C tìm nguyên nhân.
- [ ] **C8 – Script vẽ biểu đồ** (N12–N13) · file: `scripts/plot.py` · cần: C7
  Đọc `results/raw.csv`, lấy trung vị, vẽ 5 biểu đồ số liệu ở mục 8.
  Xong khi: chạy một lệnh ra đủ hình từ số liệu sơ bộ.
- [ ] **C9 – Nháp chương 1 và phần thiết lập của chương 7** (N14) · file: `report/`
  Ghi cấu hình máy đo: CPU, số nhân vật lý và luồng logic, cache, RAM, hệ điều hành, phiên bản GCC, cờ build.
  Xong khi: có bản nháp đầy đủ ý.

**Mốc M2 (cuối N14):** V2 đúng từng bit, có số đo và biểu đồ nháp. Từ đây nhóm đã có bản đủ để nộp.

### Tuần 3 (N15–N21): hoàn thiện, đo đạc, báo cáo

**Role A**

- [ ] **A11 – Gỡ lỗi V3 cùng B** (N15–N17) · cần: B5, A8
  Chạy `test_dp` và `test_correctness` với V3 trên nhiều số thread; thu hẹp trường hợp sai (W, H, thread nhỏ nhất gây lỗi) rồi chuyển cho B. Chạy ThreadSanitizer cho V3.
  Xong khi: V3 qua toàn bộ test, hoặc đến hết N17.
- [ ] **A12 – README và dọn code** (N16–N17) · file: `README.md`
  Cách cài, build, chạy, chạy test, tái lập thí nghiệm. Xoá code thừa, thống nhất định dạng.
  Xong khi: một bạn khác làm theo README từ đầu và build, chạy được.
- [ ] **A13 – Chương 6, hoàn thiện chương 2 và 3** (N18–N19) · file: `report/`
  Chương 6: phương pháp so từng bit, ma trận test, kết quả ThreadSanitizer.
  Xong khi: 3 chương ở bản cuối.
- [ ] **A14 – Slide** (N18–N20) · file: `report/` · cần: C11
  12–15 slide: bài toán, thuật toán, phân tích phụ thuộc, V1 → V2 → V3, kết quả, thảo luận, demo. B và C cung cấp hình.
  Xong khi: cả nhóm đã duyệt.

**Role B**

- [ ] **B7 – Hoàn thành V3** (N15–N17) · file: `src/omp_triangle.cpp` · cần: B5
  Bật song song cho 2 vòng; sửa lỗi A tìm ra; thử `s` = 16, 32, 45, 64 và chọn giá trị nhanh nhất.
  Xong khi: qua toàn bộ test với mọi số thread; có bảng thời gian theo `s`.
  **Hạn chót: hết N17.** Chưa xong thì dừng code, chương 5 trình bày V3 ở mức thiết kế và nêu rõ chưa hoàn thành.
- [ ] **B8 – Chương 5, hoàn thiện chương 4** (N18–N19) · file: `report/`
  Lý do của từng bước V1 → V2 → V3, kèm hình minh hoạ tam giác và đoạn code OpenMP quan trọng.
  Xong khi: 2 chương ở bản cuối.
- [ ] **B9 – Hình cho slide, đọc chéo báo cáo** (N20) · cần: A14, C13
  Xong khi: đã gửi góp ý cho chương 7 và 8.

**Role C**

- [ ] **C10 – Chạy thí nghiệm chính thức** (N16–N17, chạy lại sau khi đóng băng code) · cần: C5
  Toàn bộ ma trận ở mục 8. Máy cắm sạc, tắt ứng dụng nền, đặt `OMP_PROC_BIND=true`.
  Xong khi: `results/raw.csv` đủ mọi cấu hình, đã lưu kèm mã commit dùng để đo.
- [ ] **C11 – Biểu đồ cuối** (N18) · cần: C10
  Xong khi: đủ 6 hình ở mục 8, đã gửi cho A và B.
- [ ] **C12 – Chương 7, 8, 9** (N18–N19) · file: `report/` · cần: C11
  Kết quả, bảng Karp–Flatt, so với dự đoán Amdahl; thảo luận vì sao speedup bão hoà; so với số liệu trong tài liệu.
  Xong khi: 3 chương ở bản cuối.
- [ ] **C13 – Ghép báo cáo** (N19–N20)
  Gộp các chương, thống nhất định dạng, đánh số hình và bảng, tài liệu tham khảo, phụ lục phân công.
  Xong khi: có file báo cáo hoàn chỉnh để nhóm đọc lần cuối.

**Cả nhóm**

- [ ] **S3 – Đóng băng code** (hết N17): từ đây chỉ sửa lỗi, không thêm tính năng.
- [ ] **S4 – Chuẩn bị demo** (N20): một ảnh, chạy V0 rồi V2/V3, cho xem ảnh vẽ các seam và thời gian.
- [ ] **S5 – Tập thuyết trình** (N20, N21): 2 lần, hỏi nhau các câu ở mục 11.
- [ ] **S6 – Nộp bài** (N21).

**Việc tuỳ chọn** (chỉ làm khi task bắt buộc của mình đã xong):

- [ ] **A-opt – V4 incremental energy** · file: `src/energy_incremental.cpp`
  Sau khi xoá seam chỉ tính lại energy ở các cột `seam[i] − 1` và `seam[i]` của mỗi hàng. Phải vẫn giống từng bit với V0.
- [ ] **C-opt – Đo trên máy thứ hai** để so sánh.

---

## 8. Thiết kế thí nghiệm

**Biến độc lập:**

- Phiên bản: V0, V1, V2, V3.
- Số thread: 1, 2, 4, 8, đến số luồng logic của máy.
- Kích thước ảnh: 1280×720, 1920×1080, 3840×2160.
- Số seam xoá: 10% chiều rộng.
- Schedule (`static` / `dynamic` / `guided`): chỉ thử cho V2 ở ảnh 1080p.

**Chỉ số:**

- Thời gian tổng và từng pha.
- Speedup `S(p) = T_V0 / T_p`, so với V0 chứ không phải bản song song chạy 1 thread.
- Efficiency `E(p) = S(p) / p`.
- Karp–Flatt: `e = (1/S − 1/p) / (1 − 1/p)`.
- So với dự đoán Amdahl từ breakdown của V0.

**Quy trình đo:** cùng cờ build cho mọi phiên bản; mỗi cấu hình chạy 5 lần, lấy trung vị; không tính thời gian đọc/ghi ảnh.

**Biểu đồ:**

1. Breakdown thời gian từng pha của V0.
2. Speedup theo số thread, mỗi đường một phiên bản, có đường lý tưởng.
3. Efficiency theo số thread.
4. Speedup theo kích thước ảnh ở số thread cố định.
5. So sánh các schedule.
6. Ảnh minh hoạ: gốc, energy map, các seam, kết quả.

---

## 9. Dàn ý báo cáo

1. Giới thiệu: bài toán, động lực, mục tiêu.
2. Cơ sở lý thuyết: seam carving, quy hoạch động, OpenMP.
3. Cài đặt tuần tự, độ phức tạp, breakdown thời gian.
4. Phân tích khả năng song song: đồ thị phụ thuộc, dự đoán Amdahl.
5. Thiết kế song song: V1 → V2 → V3, lý do từng bước.
6. Kiểm chứng tính đúng.
7. Thực nghiệm: thiết lập, kết quả, biểu đồ.
8. Thảo luận: vì sao speedup bão hoà; so với kết quả trong tài liệu.
9. Kết luận và hướng phát triển (GPU, multi-seam, forward energy, video).
10. Tài liệu tham khảo; phụ lục phân công.

---

## 10. Tài liệu tham khảo

Ngoài paper gốc, các mục dưới đây được tìm qua web và mới xem tóm tắt. **Cần mở bản gốc kiểm tra trước khi trích dẫn số liệu.**

**Nền tảng thuật toán**

- Avidan, Shamir. *Seam Carving for Content-Aware Image Resizing*. SIGGRAPH 2007. https://cs.brown.edu/courses/cs016/static/files/docs/seamcarving_original_paper.pdf
- Rubinstein, Shamir, Avidan. *Improved Seam Carving for Video Retargeting*. ACM TOG 27(3), SIGGRAPH 2008. Giới thiệu forward energy. https://www.eng.tau.ac.il/~avidan/papers/vidret.pdf

**Song song hoá trên CPU (liên quan trực tiếp nhất)**

- Sam Westrick. *Parallel Seam Carving* (blog, 2020). Triangle decomposition cho DP; nêu hai giới hạn là memory-bound và span cao. **Nguồn chính cho V3, B đọc ở task B1.** https://shwestrick.github.io/2020/07/29/seam-carve.html
- *An Analysis and Implementation of Seam Carving for Content-Aware Image Resizing*. arXiv:2608.04329. Cài đặt C++17 có OpenMP tuỳ chọn; lưu ý vòng lặp trong của DP cần barrier sau mỗi hàng. https://arxiv.org/pdf/2608.04329
- *Analysis of Different Algorithmic Design Techniques for Seam Carving*. arXiv:2410.21207. https://arxiv.org/pdf/2410.21207
- Bài hội nghị (Atlantis Press) so sánh CUDA, MPI, OpenMP cho content-aware resizing: OpenMP và MPI gần như không tăng tốc. Dùng để đối chiếu kết quả ở chương 8. https://www.atlantis-press.com/article/125995760.pdf
- *Seam Carving Speed Improvement by Odd and Even Subimages Decomposition*. https://rimag.ir/en/Article/28294
- Jacob Stultz. Báo cáo project MIT 18.337 (2008) về seam carving song song. https://courses.csail.mit.edu/18.337/2008/projects/reports/stultz-6338.pdf
- *Parallelizing A Content-Aware Image Resizing with OpenSHMEM* (luận văn MTSU). https://jewlscholar.mtsu.edu/handle/mtsu/6262

**GPU (cho phần liên quan và hướng phát triển)**

- Duarte, Sendag. *Run-time Image and Video Resizing Using CUDA-enabled GPUs* (2012). https://www.ele.uri.edu/faculty/sendag/papers/shaw2012.pdf
- Kim, Zhai, Li, Chen. *Optimizing Seam Carving on multi-GPU systems for real-time image resizing*. ICPADS 2014. https://pacman.cs.tsinghua.edu.cn/~cwg/publication/icpads14/
- Chiang, Wang, Chen, Lai. *Fast JND-Based Video Carving with GPU Acceleration*. IEEE TCSVT 2009. https://graphics.cmlab.csie.ntu.edu.tw/~yiling/pdf/TCSVT.pdf
- Project CMU 15-418 *Parallelized Seam Carving* (CUDA). https://michfeng.github.io/seam-carving-site

**OpenMP và lý thuyết song song**

- OpenMP API Specification. https://www.openmp.org/specifications/
- Pacheco. *An Introduction to Parallel Programming*.
- Chapman, Jost, van der Pas. *Using OpenMP*.

**Thư viện**

- stb (đọc/ghi ảnh, header-only). https://github.com/nothings/stb

---

## 11. Rủi ro và câu hỏi phản biện

| Rủi ro | Cách xử lý |
|---|---|
| Lịch 3 tuần quá sát | V2 xong ở M2 là bản nộp tối thiểu; V3 có hạn chót N17; N21 để dự phòng |
| A2 (interface) trễ làm B và C bị chặn | A ưu tiên A2 trước mọi việc khác; nếu đến N3 chưa xong thì dùng nguyên đề xuất ở mục 4 |
| Speedup thấp | Phù hợp với tài liệu. Giải thích bằng breakdown, Karp–Flatt, băng thông bộ nhớ |
| V1 chậm hơn V0 | Giữ làm kết quả, minh chứng cho overhead |
| Kết quả song song khác tuần tự | Kiểm tra phá hoà trong argmin và biên các khối tam giác; dùng `test_dp` để thu hẹp |
| Data race | ThreadSanitizer; test từng bit, lặp nhiều lần, nhiều số thread |
| Số đo nhiễu trên laptop | Trung vị 5 lần, cắm sạc, ghi điều kiện đo |
| Lỗi môi trường build | Thống nhất GCC ngay N1; ai lỗi thì chuyển sang WSL |

Câu hỏi giảng viên có thể hỏi:

- Vì sao không song song vòng lặp xoá k seam?
- Vì sao speedup không tuyến tính? Phần tuần tự chiếm bao nhiêu?
- Vì sao chọn `static` thay vì `dynamic`?
- False sharing xảy ra ở đâu, tránh thế nào?
- Baseline có công bằng không?
- Chứng minh kết quả song song giống tuần tự thế nào?
- Triangle decomposition đúng vì sao?

---

## 12. Kiểm chứng cuối cùng trước khi nộp

- [ ] `cmake -B build -G "MinGW Makefiles" -DCMAKE_BUILD_TYPE=Release` và `cmake --build build` chạy được trên máy cả 3 người.
- [ ] `ctest --test-dir build` qua toàn bộ.
- [ ] Mọi phiên bản cho ảnh giống từng bit với V0 trên 3 ảnh × số thread 1, 2, 3, 4, 8.
- [ ] ThreadSanitizer không cảnh báo.
- [ ] `scripts/run_benchmark.ps1` rồi `scripts/plot.py` sinh lại được đủ biểu đồ trong báo cáo.
- [ ] Một người làm theo README từ đầu và ra kết quả tương tự.
- [ ] Số liệu trong báo cáo và slide khớp với `results/raw.csv`.
