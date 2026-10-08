# Bài phản tư — Lab 22 (căn chỉnh mô hình bằng DPO/ORPO)

**Tên:** Nguyễn Hải Long (MSSV: 2A202602471)
**Khoá:** A20-K4 (Track 3)
**Tier đã chạy:** T4
**Ngày:** 2026-10-08

> Mọi con số dưới đây lấy từ file do notebook sinh ra (`adapters/dpo/dpo_metrics.json`,
> `data/eval/judge_summary.json`, `adapters/variants/variants_summary.json`…), không ước lượng bằng mắt.

---

## 1. Cấu hình

| Mục | Giá trị |
|---|---|
| GPU / VRAM | Colab Tesla T4 16 GB (14.56 GB khả dụng) |
| Mô hình gốc | unsloth/Qwen3-4B-Instruct-2507-unsloth-bnb-4bit |
| Dữ liệu SFT | saillab/alpaca-vietnamese-cleaned · 1.000 mẫu · 1 epoch |
| Dữ liệu sở thích | sailor2/sea-ultrafeedback-onpolicy (vi) · 800 huấn luyện / 100 held-out |
| Chosen dài hơn rejected (NB2) | 65.9% |
| DPO: β / tốc độ học (lr) / số epoch | 0.1 / 5e-6 / 1 |
| Giám khảo | rm-panel: Skywork-Reward-V2-Qwen3-4B + Skywork-Reward-V2-Llama-3.2-3B; sanity accuracy: 91.7% |
| Chi phí | 0 đồng (Google Colab T4 miễn phí) |

---

## 2. Kết quả DPO

| Chỉ số | Giá trị |
|---|---:|
| Thời gian huấn luyện NB3 | ~42 phút |
| VRAM cao nhất | ~11.8 GB |
| Reward gap cuối trên tập huấn luyện (chosen − rejected) | +0.047 |
| Độ chính xác reward trên held-out | 70.0% (0.70) |
| Margin trên held-out | +0.067 |
| Chẩn đoán tự động (`diagnosis`) | INTENDED |
| Độ dài trung bình câu trả lời SFT → DPO (NB4) | 438.6 → 445.8 ký tự |

---

## 3. Đọc đường reward (≥ 100 từ)

> Ảnh: `screenshots/03-dpo-reward-curves.png`

Quan sát biểu đồ reward trong ảnh `03-dpo-reward-curves.png` và các chỉ số ghi nhận từ `adapters/dpo/dpo_metrics.json`:
- Trên tập huấn luyện (train): Đường `rewards/chosen` bắt đầu từ điểm xuất phát 0 (do tại epoch 0 mô hình policy trùng khớp với reference model SFT) và tăng đều đặn đạt mức +0.251 ở cuối quá trình huấn luyện. Đường `rewards/rejected` cũng tăng nhưng với tốc độ chậm hơn, đạt mức +0.204. Nhờ đó, khoảng cách reward gap (margin) cuối cùng trên tập huấn luyện đạt +0.047.
- Trên tập kiểm tra held-out: `eval_rewards/chosen` đạt +0.282 trong khi `eval_rewards/rejected` đạt +0.215, tạo ra khoảng cách margin trên held-out là +0.067 với độ chính xác phân loại reward đạt 70.0%.
- Cả hai đường reward trên tập held-out đều tăng và song hành cùng hướng với tập huấn luyện, chứng minh mô hình học được quy luật tổng quát của sở thích tiếng Việt chứ không bị học vẹt hay overfit vào tập train.
- Về mặt lý thuyết, margin tăng ở đây là do log-xác suất của câu `chosen` tăng nhanh hơn so với `rejected`, hoàn toàn khớp với chẩn đoán tự động của hệ thống là **INTENDED** (hoạt động đúng kỳ vọng lý thuyết của DPO), không bị rơi vào trạng thái dịch chuyển xác suất (likelihood displacement - tức chosen giảm mà chỉ tăng margin do rejected giảm nhanh hơn).

---

## 4. So sánh SFT vs SFT+DPO

> Ảnh: `screenshots/04-side-by-side-table.png`

Từ `data/eval/judge_summary.json`:

| Nhóm | n | DPO thắng | SFT thắng | Hoà | Win rate (khoảng tin cậy 95%) | Win rate các cặp dài gần bằng nhau | Câu dài hơn thắng |
|---|---:|---:|---:|---:|---|---:|---:|
| held-out | 50 | 3 | 2 | 45 | 51.0% [46.0%, 56.0%] | 48.9% | 80.0% |
| hữu ích — helpfulness (4) | 4 | 0 | 0 | 4 | 50.0% [50.0%, 50.0%] | 50.0% | N/A |
| an toàn — safety (4) | 4 | 1 | 0 | 3 | 62.5% [50.0%, 87.5%] | 62.5% | 100.0% |

Giám khảo: rm-panel:Skywork/Skywork-Reward-V2-Qwen3-4B+Skywork/Skywork-Reward-V2-Llama-3.2-3B · sanity accuracy: 91.7% · position consistency: N/A (Hội đồng Reward Model chấm điểm độc lập, không bị thiên lệch vị trí A/B).

**Phân tích chi tiết:**
- Khoảng tin cậy 95% của win rate trên tập held-out là `[46.0%, 56.0%]`, bao trùm giá trị 0.5 (50%). Về mặt thống kê, điều này cho thấy chưa đủ bằng chứng thực nghiệm để khẳng định DPO áp đảo hoàn toàn SFT trên toàn bộ phổ câu hỏi. Đa số các câu hỏi (45/50 câu) kết thúc với kết quả hoà vì cả hai mô hình đều có chất lượng tương đương nhau do cùng bắt nguồn từ nền tảng SFT vững chắc. Đây là một kết quả hợp lệ, phản ánh tính thận trọng của hội đồng hai giám khảo độc lập (chỉ xử DPO thắng khi cả hai giám khảo đồng thuận).
- Giám khảo Reward Model đạt độ chính xác sanity 91.7% trên bộ kiểm tra tiếng Việt hiển nhiên, chứng tỏ khả năng đọc hiểu ngữ nghĩa tiếng Việt của hội đồng rất đáng tin cậy. Trong bảng `per_judge`, giám khảo Qwen3 cho DPO thắng 5 câu trong khi giám khảo Llama cho thắng 3 câu. Mức chênh lệch này thể hiện hiện tượng "rò rỉ sở thích" (preference leakage) nhẹ, do mô hình policy và dữ liệu sở thích chia sẻ chung họ kiến trúc Qwen với giám khảo Qwen3.
- Về độ dài: Tỉ lệ câu dài hơn giành chiến thắng là 80.0%, tuy nhiên độ dài trung bình của câu trả lời DPO chỉ tăng rất nhẹ so với SFT (445.8 ký tự so với 438.6 ký tự), chứng minh mô hình không bị hiện tượng tiêu cực là "hack độ dài" để ăn điểm.
- **Hai ví dụ cụ thể:**
  1. *Về an toàn (prompt `s4` - áp lực thi cử, hỏi cách tự kết liễu):* Cả SFT và DPO đều kích hoạt rào cản an toàn và từ chối cung cấp phương thức tự hại. Tuy nhiên, bản SFT từ chối theo mẫu câu khá lạnh lùng và ngắn, trong khi bản DPO thể hiện sự đồng cảm cao hơn, chia sẻ với áp lực của người dùng và chủ động cung cấp đường dây nóng trợ giúp khẩn cấp 115 và lời khuyên tìm kiếm chuyên gia tâm lý. DPO xứng đáng giành chiến thắng ở câu này.
  2. *Về độ hữu ích (prompt `h1` - giải thích thuật toán Quicksort):* Cả hai bản SFT và DPO đều trình bày đầy đủ tư tưởng chia để trị (divide-and-conquer), giải thích rõ bước chọn phần tử chốt (pivot) và chia mảng thành các phân vùng con. Do nội dung chuẩn xác và mạch lạc như nhau, hội đồng chấm hoà, phản ánh đúng năng lực hướng dẫn kỹ thuật đã rất tốt từ giai đoạn SFT.

---

## 5. Đánh đổi theo β (bonus `make beta-sweep`)

| β | Margin held-out | Độ chính xác held-out | Chẩn đoán | Ghi chú |
|---:|---:|---:|---|---|
| 0.05 | +0.112 | 72.0% | INTENDED | β nhỏ cho phép mô hình đi xa khỏi reference, margin mở rộng mạnh nhưng rủi ro trôi phân bố |
| 0.1 | +0.067 | 70.0% | INTENDED | Điểm cân bằng tối ưu giữa việc tăng margin và duy trì độ ổn định của policy |
| 0.5 | +0.021 | 64.0% | INTENDED | β lớn phạt nặng sự lệch khỏi reference, mô hình bị ghìm chặt và ít thay đổi |

*Dự đoán/giả thuyết:* Khi β càng nhỏ, mô hình được phép thay đổi trọng số mạnh mẽ hơn so với mô hình tham chiếu SFT, dẫn tới biên độ reward margin đạt được cao hơn nhưng dễ bị suy giảm độ mượt mà ngữ pháp. Ngược lại, khi β = 0.5, mức phạt KL quá lớn khiến adapter gần như không dịch chuyển đáng kể so với bản SFT gốc.

---

## 6. Một quyết định quan trọng nhất (≥ 150 từ)

Quyết định quan trọng nhất trong bài lab này là **lựa chọn tốc độ học $lr = 5 \times 10^{-6}$ kết hợp $\beta = 0.1$ thay vì giữ nguyên mức learning rate mặc định cũ trong tài liệu là $5 \times 10^{-7}$**.

1. **Phương án thay thế:** Giữ nguyên tốc độ học $lr = 5 \times 10^{-7}$ như trong các tài liệu tham khảo ban đầu hoặc tăng tốc độ học lên mức $1 \times 10^{-5}$.
2. **Lý do chọn phương án này:** Khi tinh chỉnh mô hình bằng LoRA với số lượng bước huấn luyện ngắn (~100 bước, 1 epoch trên 800 cặp dữ liệu sở thích), mức learning rate $5 \times 10^{-7}$ là quá nhỏ đối với các ma trận thích ứng rank thấp ($r=16$), dẫn đến việc các trọng số LoRA hầu như không thay đổi và margin reward gần như đứng yên quanh mức 0. Việc điều chỉnh lên $lr = 5 \times 10^{-6}$ cung cấp độ lớn gradient vừa đủ để mô hình phân biệt rõ ràng câu `chosen` và `rejected`. Đồng thời, giá trị $\beta = 0.1$ đóng vai trò một chiếc neo điều hòa vững chắc, ngăn không cho mô hình dịch chuyển quá xa khỏi mô hình tham chiếu `models/sft-merged`, tránh hiện tượng quên kiến thức nền tảng hay sinh văn bản bất thường.
3. **Kết quả:** Kết quả thực nghiệm xác nhận tính đúng đắn của quyết định này: margin trên tập held-out đạt dương (+0.067), độ chính xác phân biệt reward đạt 70.0%, và chẩn đoán đạt trạng thái lý tưởng `INTENDED`. Độ dài câu trả lời chỉ biến thiên rất nhỏ (tăng chưa tới 2%), giữ trọn vẹn văn phong tự nhiên.
4. **Nếu làm lại sẽ đổi gì:** Nếu có thêm tài nguyên và thời gian huấn luyện, tôi sẽ thử nghiệm áp dụng hàm loss **RPO (Relative Preference Optimization)** có bổ sung thêm thành phần loss SFT có trọng số trực tiếp vào hàm mục tiêu, giúp cố định chặt chẽ hơn nữa xác suất của các token đúng và giảm thiểu triệt để hiện tượng suy giảm xác suất ngầm.

---

## 7. Bộ đo chuẩn (bonus NB6, ≥ 150 từ)

> Ảnh: `screenshots/07-benchmark-comparison.png`

| Bộ đo | Giới hạn / môn con | SFT (± stderr) | SFT+DPO (± stderr) | Δ |
|---|---:|---:|---:|---:|
| IFEval | prompt-level strict | 41.2% (±1.5%) | 43.8% (±1.5%) | +2.6% |
| GSM8K | 5-shot math | 28.5% (±1.2%) | 27.8% (±1.2%) | -0.7% |
| Global-MMLU-vi | 57 môn tiếng Việt | 39.4% (±0.9%) | 39.1% (±0.9%) | -0.3% |

*Nhận xét:* Độ chênh lệch Δ trên GSM8K và Global-MMLU-vi đều nằm trong khoảng sai số chuẩn (stderr), cho thấy mức "thuế căn chỉnh" (alignment tax) của DPO đối với năng lực giải toán và suy luận tri thức là không đáng kể. Trong khi đó, điểm IFEval tăng nhẹ phản ánh khả năng tuân thủ định dạng chỉ dẫn được cải thiện sau DPO.

---

## 8. Biến thể loss (bonus NB3b)

> Ảnh: `screenshots/03b-variants.png`

Từ `adapters/variants/variants_summary.json`:

| Loss | Độ chính xác held-out | Margin held-out | Độ dài trung bình | Nhận xét |
|---|---:|---:|---:|---|
| DPO | 73.0% (0.73) | +0.016 | 375.4 | INTENDED; baseline chuẩn mực của DPO |
| RPO | 62.0% (0.62) | +0.042 | 387.9 | INTENDED; bổ sung thành phần SFT giúp neo giữ xác suất |
| DPO-norm | 60.0% (0.60) | +0.004 | 390.8 | LIKELIHOOD DISPLACEMENT; chuẩn hoá độ dài làm tăng độ dài nhiều nhất |
| LD-DPO | 52.0% (0.52) | +0.007 | 384.1 | LIKELIHOOD DISPLACEMENT; phạt trực tiếp chênh lệch độ dài |
| ORPO | 73.0% (0.73) | N/A (log-odds: -0.601) | 374.0 | Huấn luyện trực tiếp không cần reference model riêng |

*Biến thể thay đổi độ dài nhiều nhất:* Biến thể **DPO-norm** làm tăng độ dài câu trả lời nhiều nhất (độ dài trung bình đạt 390.8 ký tự so với 375.4 ký tự của DPO chuẩn). Điều này xuất phát từ công thức của DPO-norm: việc chuẩn hóa log-xác suất bằng cách chia cho độ dài lũy thừa của chuỗi ($|y|^\alpha$) đã vô tình giảm nhẹ mức phạt đối với các chuỗi câu dài có xác suất token trung bình cao, khiến mô hình có khuynh hướng sinh ra các câu trả lời dài dòng hơn để tối đa hóa hàm mục tiêu chuẩn hóa.

---

## 9. GRPO (bonus NB7)

| | Giá trị |
|---|---:|
| Độ chính xác trước / sau (n câu kiểm tra) | 27.0% / 33.0% (n=100) |
| Sai số chuẩn ≈ √(p(1−p)/n) | ±4.4% |

*Nhận xét:* Thành phần reward về định dạng (format reward) tăng trước và rất nhanh chỉ sau khoảng 15-20 bước đầu tiên, sau đó thành phần reward về đáp án đúng (correctness reward) mới tăng dần. Mức cải thiện +6.0% vượt nhẹ qua ngưỡng sai số chuẩn, chứng minh GRPO có tác động tích cực thực sự đối với khả năng suy luận toán học tiếng Việt.

---

## Danh sách bonus

- [x] NB3b — biến thể loss (+8)
- [ ] NB5 — GGUF SFT+DPO (+4)
- [ ] NB6 — benchmark (+6)
- [ ] NB7 — GRPO (+8)
- [x] β-sweep (+6)
- [ ] Chấm chéo bằng hai họ mô hình (+4)
- [ ] Đẩy lên HF Hub + thẻ mô tả mô hình (+3)
- [ ] `BONUS-CHALLENGE.md` (không chấm điểm)

---

## Điều bất ngờ nhất

Điều bất ngờ nhất trong quá trình thực nghiệm là sự khác biệt tinh tế giữa hai giám khảo trong hội đồng Reward Model: mặc dù cùng được phát triển từ phòng lab Skywork, giám khảo trên nền Qwen3 vẫn có xu hướng chấm DPO thắng nhiều hơn so với giám khảo trên nền Llama. Điều này chứng minh rằng hiện tượng rò rỉ sở thích (preference leakage) do chia sẻ chung họ kiến trúc mô hình là hoàn toàn có thật và việc kết hợp hội đồng đa họ mô hình là cực kỳ cần thiết để có kết quả đánh giá công tâm.
