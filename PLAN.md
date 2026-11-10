# Plan: Parallel Seam Carving bằng OpenMP (nhóm 3 người, 3 tuần)

## 1. Context

Bài tập nhóm cuối kì môn Tính toán song song: cài đặt song song thuật toán **Seam Carving** (Avidan & Shamir, SIGGRAPH 2007) để co ảnh theo nội dung, dùng **OpenMP**. Thời gian: **3 tuần**. Thư mục dự án hiện có `PLAN.md` và `seamcarving_original_paper.pdf`, chưa có code.

Môn học chấm phần *song song hoá*, không chấm xử lý ảnh. Trọng tâm:

1. Bản tuần tự đúng, làm baseline.
2. Phân tích phụ thuộc dữ liệu: chỗ nào song song được, vì sao.
3. Các phiên bản OpenMP từ ngây thơ đến tối ưu.
4. Đo đạc nghiêm túc (speedup, efficiency, Amdahl, Karp–Flatt) và giải thích được kết quả, kể cả khi speedup thấp.

**Kỳ vọng thực tế:** các tài liệu tìm được cho thấy OpenMP trên seam carving chuẩn chỉ cho speedup khiêm tốn, có bài báo cáo gần như không tăng tốc (mục 9). Lý do: bước quy hoạch động phụ thuộc theo hàng và bài toán bị giới hạn bởi băng thông bộ nhớ. Vì vậy điểm mạnh của bài phải nằm ở phần phân tích, không phải ở con số speedup.

**Giả định (sửa nếu sai):**

| Hạng mục | Giả định |
|---|---|
| Ngôn ngữ | C++17 + OpenMP |
| Sản phẩm nộp | Mã nguồn, báo cáo, slide, demo |
| Compiler | GCC qua MSYS2/MinGW-w64 hoặc WSL. **Không dùng MSVC mặc định**: `/openmp` chỉ hỗ trợ OpenMP 2.0, thiếu `reduction(min:)`, `collapse`, `simd` |

---

## 2. Tóm tắt thuật toán và phân tích song song

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

Phạm vi đã cắt cho 3 tuần. Tất cả dùng chung một interface, chọn bằng tham số dòng lệnh.

| Phiên bản | Mức | Nội dung | Mục đích |
|---|---|---|---|
| **V0 – Sequential** | Bắt buộc | Tuần tự, tối ưu hợp lý (mảng 1D liên tục, `-O2`) | Baseline tính speedup |
| **V1 – Naive OpenMP** | Bắt buộc | `#pragma omp parallel for` cho energy, từng hàng DP, xoá seam; mỗi hàng DP mở một parallel region | Cho thấy overhead fork/join |
| **V2 – Tuned OpenMP** | Bắt buộc, bản chính | Một `parallel` region bao toàn bộ DP, `omp for` từng hàng; `schedule(static)`; argmin bằng reduction; `simd` cho energy | Bản nộp tối thiểu |
| **V3 – Triangle decomposition** | Mục tiêu, có hạn chót | Chia DP thành các dải nhiều hàng; mỗi dải chạy 2 vòng song song: tam giác xuôi, rồi tam giác ngược lấp chỗ trống. Đáy tam giác chẵn để các khối khít nhau; tài liệu gợi ý đáy khoảng 64–90 pixel | Tăng độ hạt, giảm số lần đồng bộ |
| **V4 – Incremental energy** | Tuỳ chọn | Sau khi xoá seam chỉ tính lại energy quanh seam | Thay đổi tỉ lệ các pha, phân tích Amdahl mới |

**Hạn chót cho V3:** hết ngày thứ 3 của tuần 3. Nếu chưa đúng từng bit thì dừng, trình bày V3 ở mức thiết kế trong báo cáo.

**Đã bỏ khỏi phạm vi:** forward energy, seam ngang, xoá nhiều seam mỗi vòng, phóng to ảnh. Chỉ nhắc ở phần "hướng phát triển".

**Yêu cầu tính đúng:** V1–V4 phải cho ảnh **giống từng bit** với V0. Quy tắc phá hoà phải xác định: khi bằng nhau luôn chọn cột trái nhất, kể cả trong argmin song song.

---

## 4. Cấu trúc project

```
TTSS - CK/
├── PLAN.md
├── CMakeLists.txt
├── README.md                 # cách build, chạy, tái lập thí nghiệm
├── docs/                     # seamcarving_original_paper.pdf và tài liệu khác
├── third_party/              # stb_image.h, stb_image_write.h
├── include/
│   ├── image.hpp             # struct Image (mảng 1D), load/save
│   ├── seam_carving.hpp      # interface chung: energy, dp, find_seam, remove_seam
│   └── timer.hpp             # bọc omp_get_wtime, cộng dồn theo pha
├── src/
│   ├── image.cpp
│   ├── seq.cpp               # V0
│   ├── omp_naive.cpp         # V1
│   ├── omp_tuned.cpp         # V2
│   ├── omp_triangle.cpp      # V3
│   └── main.cpp              # CLI
├── tests/test_correctness.cpp
├── data/                     # ảnh test
├── scripts/
│   ├── run_benchmark.ps1     # quét version × threads × ảnh, ghi CSV
│   └── plot.py               # matplotlib
├── results/                  # CSV + hình
└── report/                   # báo cáo + slide
```

CLI:

```
seamcarve --input a.jpg --output b.png --remove-cols 300 --version tuned --threads 8 --timing
```

`--timing` in thời gian từng pha (energy / dp / find / remove / tổng) dạng CSV.

---

## 5. Phân công

| Thành viên | Vai trò | Phần code | Phần báo cáo |
|---|---|---|---|
| **A – Thuật toán & baseline** | Trưởng nhóm kỹ thuật | `image`, V0, CLI, `tests/`; V4 nếu kịp | Cơ sở lý thuyết, thiết kế tuần tự, kiểm chứng tính đúng |
| **B – Song song hoá DP** | Phần khó nhất | DP của V1, V2, V3; argmin song song | Phân tích phụ thuộc, thiết kế song song, triangle decomposition |
| **C – Benchmark & phân tích** | Thực nghiệm, điều phối báo cáo | Energy và remove song song (V1, V2), `timer`, `scripts/` | Thiết lập thí nghiệm, kết quả, biểu đồ, Amdahl/Karp–Flatt, slide |

Quy tắc làm việc:

- Dùng Git, mỗi người một nhánh, merge sau khi một người khác xem.
- A chốt `seam_carving.hpp` trong **3 ngày đầu**, để B và C không bị chặn.
- Họp nhanh 15 phút, 3 lần mỗi tuần (lịch gấp nên cần gặp dày hơn).
- Báo cáo viết song song với code từ tuần 2, không dồn về cuối.
- Mỗi người phải hiểu toàn bộ code để trả lời vấn đáp.

---

## 6. Lộ trình 3 tuần

### Tuần 1 – Baseline và bản ngây thơ

| | Ngày 1–3 | Ngày 4–7 |
|---|---|---|
| **Cả nhóm** | Đọc paper gốc mục 3; cài GCC + CMake; chạy thử OpenMP hello | |
| **A** | `Image` + đọc/ghi bằng stb; chốt interface | Xong V0; test trên ma trận nhỏ tính tay; test so sánh từng bit |
| **B** | Vẽ đồ thị phụ thuộc DP; đọc bài blog về triangle decomposition (mục 9) | DP của V1; nháp thiết kế V2, V3 |
| **C** | Khung CMake, `timer`, thu thập ảnh test | Energy và remove của V1; script benchmark ghi CSV |

**Mốc cuối tuần 1:** V0 và V1 chạy đúng; có breakdown thời gian từng pha của V0 (đầu vào cho Amdahl).

### Tuần 2 – Bản chính và V3

| | Ngày 8–10 | Ngày 11–14 |
|---|---|---|
| **A** | Hoàn thiện test (nhiều ảnh × nhiều số thread, kể cả số lẻ) | ThreadSanitizer trên WSL; viết báo cáo chương 2, 3; V4 nếu rảnh |
| **B** | V2: một parallel region, argmin reduction | V3: cài tam giác xuôi/ngược |
| **C** | Tối ưu energy/remove của V2 (`simd`, `memmove` theo hàng) | Benchmark sơ bộ V0/V1/V2; viết `plot.py`; viết chương thiết lập thí nghiệm |

**Mốc cuối tuần 2:** V2 đúng từng bit, đã có số đo sơ bộ và biểu đồ nháp. Từ đây nhóm đã có thứ để nộp.

### Tuần 3 – Đo đạc, báo cáo, thuyết trình

| | Ngày 15–17 | Ngày 18–19 | Ngày 20–21 |
|---|---|---|---|
| **A** | Giúp B gỡ lỗi V3; viết README; dọn code | Viết chương kiểm chứng; đọc chéo | Tập thuyết trình |
| **B** | Xong V3 (hạn chót ngày 17), dò chiều cao dải | Viết chương phân tích và thiết kế song song | Tập thuyết trình |
| **C** | Chạy toàn bộ ma trận thí nghiệm (chạy lại khi V3 xong) | Biểu đồ cuối; chương kết quả và thảo luận; slide | Tập thuyết trình |

- **Ngày 17:** đóng băng code.
- **Ngày 19:** báo cáo xong bản đầy đủ.
- **Ngày 20–21:** sửa báo cáo, slide 12–15 trang, tập 2 lần, chuẩn bị demo (một ảnh, so V0 với V2/V3, ảnh vẽ các seam).
- Để dành ngày 21 làm dự phòng.

---

## 7. Thiết kế thí nghiệm

**Biến độc lập:**

- Phiên bản: V0, V1, V2, V3.
- Số thread: 1, 2, 4, 8, đến số luồng logic của máy. Ghi rõ số nhân vật lý.
- Kích thước ảnh: 3 mức, khoảng 1280×720, 1920×1080, 3840×2160.
- Số seam xoá: 10% chiều rộng.
- Schedule (`static` / `dynamic` / `guided`): chỉ thử cho V2 ở một kích thước ảnh.

**Chỉ số:**

- Thời gian tổng và từng pha.
- Speedup `S(p) = T_V0 / T_p`, so với V0 chứ không phải bản song song chạy 1 thread.
- Efficiency `E(p) = S(p) / p`.
- Karp–Flatt: `e = (1/S − 1/p) / (1 − 1/p)`.
- So với dự đoán Amdahl từ breakdown của V0.

**Quy trình đo:**

- Cùng cờ build cho mọi phiên bản (`-O2`).
- Mỗi cấu hình chạy 5 lần, lấy trung vị.
- Không tính thời gian đọc/ghi ảnh.
- Cắm sạc, tắt ứng dụng nền, `OMP_PROC_BIND=true`.
- Ghi cấu hình máy: CPU, số nhân, cache, RAM, hệ điều hành, phiên bản GCC.
- Đo trên một máy chính; máy thứ hai nếu còn thời gian.

**Biểu đồ:**

1. Breakdown thời gian từng pha của V0.
2. Speedup theo số thread, mỗi đường một phiên bản, có đường lý tưởng.
3. Efficiency theo số thread.
4. Speedup theo kích thước ảnh ở số thread cố định.
5. So sánh các schedule.
6. Ảnh minh hoạ: gốc, energy map, các seam, kết quả.

---

## 8. Dàn ý báo cáo

1. Giới thiệu: bài toán, động lực, mục tiêu.
2. Cơ sở lý thuyết: seam carving, quy hoạch động, OpenMP.
3. Cài đặt tuần tự, độ phức tạp, breakdown thời gian.
4. Phân tích khả năng song song: đồ thị phụ thuộc, dự đoán Amdahl.
5. Thiết kế song song: V1 → V2 → V3, lý do từng bước.
6. Kiểm chứng tính đúng.
7. Thực nghiệm: thiết lập, kết quả, biểu đồ.
8. Thảo luận: vì sao speedup bão hoà; so với kết quả trong tài liệu (mục 9).
9. Kết luận và hướng phát triển (GPU, multi-seam, forward energy, video).
10. Tài liệu tham khảo; phụ lục phân công.

---

## 9. Tài liệu tham khảo

Ngoài paper gốc, các mục dưới đây tôi tìm qua web và mới xem tóm tắt, chưa đọc toàn văn. Nhóm cần mở bản gốc kiểm tra trước khi trích dẫn số liệu.

**Nền tảng thuật toán**

- Avidan, Shamir. *Seam Carving for Content-Aware Image Resizing*. SIGGRAPH 2007. Trong dự án: `seamcarving_original_paper.pdf`.
- Rubinstein, Shamir, Avidan. *Improved Seam Carving for Video Retargeting*. ACM TOG 27(3), SIGGRAPH 2008. Giới thiệu forward energy. https://www.eng.tau.ac.il/~avidan/papers/vidret.pdf

**Song song hoá trên CPU (liên quan trực tiếp nhất)**

- Sam Westrick. *Parallel Seam Carving* (blog, 2020). Mô tả triangle decomposition cho DP: dải nhiều hàng, hai vòng tam giác xuôi/ngược; nêu hai giới hạn là memory-bound và span cao. Nguồn chính cho V3. https://shwestrick.github.io/2020/07/29/seam-carve.html
- *An Analysis and Implementation of Seam Carving for Content-Aware Image Resizing*. arXiv:2608.04329. Cài đặt C++17 có OpenMP tuỳ chọn; lưu ý vòng lặp trong của DP cần barrier sau mỗi hàng. https://arxiv.org/pdf/2608.04329
- *Analysis of Different Algorithmic Design Techniques for Seam Carving*. arXiv:2410.21207. Nêu tính độc lập giữa các cột trong một hàng DP. https://arxiv.org/pdf/2410.21207
- Bài hội nghị (Atlantis Press) so sánh CUDA, MPI, OpenMP cho content-aware resizing: OpenMP và MPI gần như không tăng tốc; phần seam map khó song song do bản chất quy hoạch động. Dùng để đối chiếu kết quả. https://www.atlantis-press.com/article/125995760.pdf
- *Seam Carving Speed Improvement by Odd and Even Subimages Decomposition*. Tìm seam song song trên các ảnh con. https://rimag.ir/en/Article/28294
- Jacob Stultz. Báo cáo project MIT 18.337 (2008) về seam carving song song. https://courses.csail.mit.edu/18.337/2008/projects/reports/stultz-6338.pdf
- *Parallelizing A Content-Aware Image Resizing with OpenSHMEM* (luận văn MTSU). https://jewlscholar.mtsu.edu/handle/mtsu/6262

**GPU (cho phần liên quan và hướng phát triển)**

- Duarte, Sendag. *Run-time Image and Video Resizing Using CUDA-enabled GPUs* (2012). https://www.ele.uri.edu/faculty/sendag/papers/shaw2012.pdf
- Kim, Zhai, Li, Chen. *Optimizing Seam Carving on multi-GPU systems for real-time image resizing*. ICPADS 2014; bản tạp chí J. Supercomputing 2015. Non-cumulative seam carving, multi-seam. https://pacman.cs.tsinghua.edu.cn/~cwg/publication/icpads14/
- Chiang, Wang, Chen, Lai. *Fast JND-Based Video Carving with GPU Acceleration*. IEEE TCSVT 2009. https://graphics.cmlab.csie.ntu.edu.tw/~yiling/pdf/TCSVT.pdf
- Project CMU 15-418 *Parallelized Seam Carving* (CUDA). https://michfeng.github.io/seam-carving-site

**OpenMP và lý thuyết song song**

- OpenMP API Specification. https://www.openmp.org/specifications/
- Pacheco. *An Introduction to Parallel Programming* (speedup, efficiency, Amdahl, chương OpenMP).
- Chapman, Jost, van der Pas. *Using OpenMP*.

**Thư viện**

- stb (đọc/ghi ảnh, header-only). https://github.com/nothings/stb

---

## 10. Rủi ro và câu hỏi phản biện

| Rủi ro | Cách xử lý |
|---|---|
| Lịch 3 tuần quá sát | V2 xong cuối tuần 2 là bản nộp tối thiểu; V3 có hạn chót ngày 17; ngày 21 để dự phòng |
| Speedup thấp (khoảng 2–3× trên 8 thread, hoặc thấp hơn) | Phù hợp với tài liệu. Giải thích bằng breakdown, Karp–Flatt, băng thông bộ nhớ |
| V1 chậm hơn V0 | Giữ làm kết quả, minh chứng cho overhead |
| Kết quả song song khác tuần tự | Kiểm tra phá hoà trong argmin và biên các khối tam giác |
| Data race | ThreadSanitizer; test từng bit, lặp nhiều lần, nhiều số thread |
| Số đo nhiễu trên laptop | Trung vị 5 lần, cắm sạc, ghi điều kiện đo |
| Lỗi môi trường build | Thống nhất GCC ngay ngày 1; ai lỗi thì chuyển sang WSL |

Câu hỏi giảng viên có thể hỏi:

- Vì sao không song song vòng lặp xoá k seam?
- Vì sao speedup không tuyến tính? Phần tuần tự chiếm bao nhiêu?
- Vì sao chọn `static` thay vì `dynamic`?
- False sharing xảy ra ở đâu, tránh thế nào?
- Baseline có công bằng không?
- Chứng minh kết quả song song giống tuần tự thế nào?
- Triangle decomposition đúng vì sao? (Vùng phụ thuộc của một ô là tam giác ngược.)

---

## 11. Kiểm chứng

1. **Build:** `cmake -B build -G "MinGW Makefiles" -DCMAKE_BUILD_TYPE=Release` rồi `cmake --build build` chạy được trên máy cả 3 người.
2. **Đúng thuật toán:** V0 đúng trên ma trận energy nhỏ tính tay; ảnh kết quả hợp lý so với hình trong paper.
3. **Đúng song song:** `tests/test_correctness` so sánh từng bit V1–V3 với V0, 3 ảnh × số thread 1, 2, 3, 4, 8, mỗi cấu hình lặp 10 lần.
4. **Không data race:** build `-fsanitize=thread` trên WSL/Linux, không cảnh báo.
5. **Hiệu năng:** `scripts/run_benchmark.ps1` sinh CSV; `scripts/plot.py` sinh đủ biểu đồ mục 7.
6. **Tái lập:** một người không viết phần đó làm theo README và ra kết quả tương tự.

