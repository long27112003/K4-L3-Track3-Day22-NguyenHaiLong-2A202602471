# Bài phản tư — Lab 22: DPO/ORPO Alignment

**Họ tên:** Nguyễn Hải Long · **MSSV:** 2A202602471

**Track:** 3 · **Ngày cập nhật:** 2026-10-24
**Môi trường thực chạy:** Windows, NVIDIA GeForce RTX 3060 12 GB, Python 3.11, tier cấu hình T4.

Các số liệu bên dưới lấy từ JSON, notebook và hình trong repository. Trọng số SFT/DPO và GGUF được giữ cục bộ vì kích thước lớn; không có trong GitHub.

## 1. Cấu hình và dữ liệu

Mô hình gốc: `unsloth/Qwen3-4B-Instruct-2507-unsloth-bnb-4bit`. SFT mini dùng `saillab/alpaca-vietnamese-cleaned` (1.000 mẫu, một epoch); mô hình đã gộp nằm tại `models/sft-merged/`. Dữ liệu ưu tiên: `sailor2/sea-ultrafeedback-onpolicy`, 800 cặp huấn luyện và 100 cặp held-out, chia theo prompt để tránh trùng. Trong tập huấn luyện, 65,875% câu `chosen` dài hơn `rejected`. DPO dùng β = 0,1, learning rate = 5e-6 và một epoch. Hai giám khảo local là Skywork Reward V2 dựa trên Qwen3-4B và Llama-3.2-3B; sanity accuracy đều bằng 100%.

## 2. Kết quả DPO

| Chỉ số | Kết quả |
|---|---:|
| Loss đầu → cuối | 0,7095 → 0,6865 |
| Reward train chosen / rejected / gap | 0,1736 / 0,1144 / +0,0592 |
| Reward held-out chosen / rejected / gap | 0,1772 / 0,1358 / +0,0414 |
| Reward accuracy held-out | 62% |
| Chẩn đoán | `INTENDED` |

## 3. Đọc đường reward

Hình `screenshots/03-dpo-reward-curves.png` tách `chosen` và `rejected` ở train và held-out. Reward ngầm ở điểm khởi đầu xấp xỉ 0 do policy vừa được khởi tạo từ SFT, trùng reference. Cuối quá trình, reward chosen train 0,1736 cao hơn rejected 0,1144; gap dương 0,0592. Ở held-out, chosen 0,1772 và rejected 0,1358, gap 0,0414. Cả hai reward đều tăng, nhưng chosen tăng nhiều hơn; đây là `INTENDED` theo hàm chẩn đoán. Vì rejected cũng tăng, không thể diễn đạt rằng DPO đã hạ tuyệt đối xác suất rejected. Kết quả held-out thấp hơn train và accuracy 62% cho thấy tín hiệu ưu tiên có chuyển sang prompt chưa thấy, song chưa đủ để khẳng định mô hình cải thiện mạnh về chất lượng sinh. Khi chỉ nhìn gap, người đọc có thể bỏ qua hiện tượng likelihood displacement: chosen giảm mà rejected giảm nhanh hơn vẫn tạo gap dương. Lần chạy này không thuộc trường hợp đó, nhưng DPO-norm và LD-DPO trong NB3b có dấu hiệu ấy. Cần kết hợp đường reward, held-out và đánh giá câu trả lời thực tế thay vì lấy một chỉ số duy nhất.

## 4. So sánh SFT và SFT+DPO

| Nhóm | n | DPO thắng | SFT thắng | Hòa | DPO win rate, CI 95% |
|---|---:|---:|---:|---:|---:|
| Held-out | 50 | 13 | 11 | 26 | 52% [43%, 62%] |
| Helpfulness | 4 | 2 | 1 | 1 | 62,5% [25%, 100%] |
| Safety | 4 | 0 | 1 | 3 | 37,5% [12,5%, 50%] |

Trên held-out, SFT dài trung bình 547,9 ký tự và DPO 585,36 ký tự. Câu dài hơn thắng 58,3% các cặp có phân thắng bại; 43 cặp có độ dài gần bằng nhau cho win rate DPO 53,5%. Khoảng tin cậy held-out chứa 50%, nên chưa phát hiện ưu thế rõ của DPO. Riêng safety chỉ có bốn prompt và DPO không thắng prompt nào, cần thận trọng, không được suy rộng thành khẳng định an toàn hơn. Hai giám khảo local có DPO win rate lần lượt 53% (Qwen3) và 50% (Llama); mức đồng ý 89,7% trên 58 prompt. Đây đều là mô hình của Skywork, cùng nhóm với reward model gắn nhãn dữ liệu; giám khảo Qwen còn cùng họ kiến trúc với policy. Điều này có thể tạo thiên lệch. Chấm chéo 58 prompt bằng OpenAI `gpt-4o-mini` (hai thứ tự A/B) cho mức đồng ý với hội đồng local 63,8%; riêng 50 held-out, API chấm DPO thắng 3, SFT thắng 3, hòa 44, win rate 50% [45%, 55%] và position consistency 86%. Mức đồng ý chưa cao củng cố việc không tuyên bố DPO vượt SFT; `position_consistency` không áp dụng cho cách chấm reward độc lập của hội đồng local.

## 5. Đánh đổi theo β

Chưa chạy `make beta-sweep`, nên không có số đo thật cho β = 0,05 và 0,5. Giá trị β = 0,1 ở lần DPO chính cho margin held-out +0,0414 và accuracy 62%, nhưng không thay thế được phép so sánh ba lần huấn luyện cùng thiết lập. Giả thuyết: β thấp có thể cho policy đi xa reference hơn, trong khi β cao hạn chế thay đổi; hướng của reward gap và chất lượng sinh phải được kiểm chứng từ từng run, không suy ra trực tiếp từ β. Khi có đủ tài nguyên GPU, cần chạy ba cấu hình, giữ cố định chia dữ liệu và seed, rồi vẽ `bonus-beta-sweep.png` từ các `dpo_metrics.json` mới.

## 6. Quyết định quan trọng nhất

Tôi dùng learning rate 5e-6 với β = 0,1 và kiểm tra cả reward train lẫn held-out trước khi kết luận. Phương án thay thế là dùng learning rate nhỏ hơn, chẳng hạn 5e-7, hoặc tăng lên 1e-5. Với một epoch và 800 cặp, mức quá nhỏ có nguy cơ cho adapter LoRA thay đổi ít; mức quá lớn có thể làm kết quả thiếu ổn định và tăng sai lệch so với reference. Tôi chọn 5e-6 như một cấu hình thực nghiệm vừa phải, không khẳng định đó là tối ưu vì chưa có sweep learning rate. Loss ghi nhận giảm từ 0,7095 xuống 0,6865; gap held-out là +0,0414 với accuracy 62%. Các số này cho thấy mô hình học được một phần tín hiệu ưu tiên. Tuy nhiên, so sánh sinh câu trả lời chỉ đạt DPO win rate 52%, khoảng tin cậy [43%, 62%] bao gồm mức hòa 50%; không thể coi đó là bằng chứng DPO vượt hẳn SFT. Chấm chéo bằng API thuộc họ khác cho mức đồng ý chỉ 63,8%, nên tôi càng cần thận trọng. Nếu làm lại, tôi sẽ chạy nhiều seed và β-sweep. Tôi cũng sẽ rà các câu safety thua, vì điểm trung bình thuận lợi không bù được lỗi ở nhóm này. Quyết định này giúp báo cáo trung thực về hiệu quả, giới hạn thống kê và rủi ro của mô hình.

## 7. Benchmark NB6

| Bộ đo | Giới hạn | SFT (± stderr) | SFT+DPO (± stderr) | Chênh lệch |
|---|---:|---:|---:|---:|
| IFEval | 200 | 49,0% ± 3,54% | 50,0% ± 3,54% | +1,0 điểm % |
| GSM8K | 250 | 82,0% ± 2,43% | 85,2% ± 2,25% | +3,2 điểm % |
| Global-MMLU-vi | 10/môn | 24,21% ± 1,79% | 24,21% ± 1,79% | 0 điểm % |

NB6 dùng chat template cho hai mô hình và tạo `screenshots/07-benchmark-comparison.png`. IFEval tăng 1 điểm phần trăm, nhỏ hơn stderr khoảng 3,54 điểm của mỗi ước lượng. Với GSM8K, tăng 3,2 điểm phần trăm nhưng stderr từng vế là 2,43 và 2,25 điểm; không đủ để kết luận chắc chắn có cải thiện nếu không phân tích chênh lệch theo cùng câu hỏi hoặc lặp nhiều seed. Global-MMLU-vi bằng nhau ở độ chính xác tổng hợp, nhưng chỉ lấy 10 mẫu mỗi môn nên dễ bị ảnh hưởng bởi chọn mẫu; điểm 24,21% không nên xem như kết quả trên toàn bộ benchmark. Hai lần đánh giá dùng cùng cấu hình prompt, seed và mô hình gốc, song khoảng sai số ở đây không phản ánh mọi nguồn biến thiên của huấn luyện. Không thấy alignment tax rõ trong ba phép đo được chọn, cũng không chứng minh DPO làm tăng năng lực chung. Một bước tiếp theo hợp lý là đánh giá toàn bộ tập hoặc tạo bootstrap trên dự đoán theo từng câu, rồi đối chiếu một nhóm prompt tiếng Việt khó và các câu safety.

## 8. Biến thể loss NB3b

| Loss | Accuracy held-out | Gap reward | Độ dài trung bình (ký tự) | Chẩn đoán |
|---|---:|---:|---:|---|
| DPO | 56% | +0,0199 | 442,90 | INTENDED |
| RPO | 54% | +0,0106 | 458,55 | INTENDED |
| DPO-norm | 57% | +0,0183 | 437,85 | LIKELIHOOD DISPLACEMENT |
| LD-DPO | 60% | +0,0118 | 447,75 | LIKELIHOOD DISPLACEMENT |
| ORPO | 65% | log-odds −0,6220 | 446,70 | Không dùng reference riêng |

Trong phép so sánh này, RPO cho câu trả lời dài nhất (458,55 ký tự), cao hơn DPO 15,65 ký tự. Đây là quan sát thực nghiệm trên số prompt đã dùng, không đủ để khẳng định RPO luôn làm tăng độ dài. DPO-norm và LD-DPO có gap dương nhưng reward chosen âm, vì vậy cần đọc cả hai vế thay vì coi margin là thành công trọn vẹn. Bảng biến thể dùng một tập chạy riêng nên không đồng nhất với độ chính xác 62% ở NB3 chính.

## 9. GGUF NB5 và GRPO NB7

NB5 xuất mô hình SFT+DPO sang Q4_K_M, file cục bộ khoảng 2.497 MB. `data/eval/deploy_meta.json` giữ prompt Bubble Sort và cả hai câu trả lời HF/GGUF. Cả hai giải thích đúng ý so sánh cặp phần tử và lặp đến khi sắp xếp xong. Tuy nhiên output có token `<tool_call>`/`</tool_call>` dư, cần tinh chỉnh template hoặc hậu xử lý trước khi dùng thực tế.

NB7 chạy 60 bước GRPO, bốn generations mỗi prompt trên `vuongtsc/vi-gsm8k-agentic`; kiểm tra 100 câu, accuracy trước 51% và sau 49%. Chênh −2 điểm phần trăm nằm trong nhiễu lấy mẫu: stderr xấp xỉ 5 điểm phần trăm ở mỗi tỷ lệ. Hình `screenshots/08-grpo-reward.png` cho thấy mean reward dao động khoảng 0,3–1,3 và kết thúc xấp xỉ 1,0; vùng `reward_std` khá rộng, không có xu thế tăng đều. Reward được cộng từ tính đúng và định dạng, nên chỉ nhìn tổng reward không xác định được thành phần nào tăng nhanh hơn hoặc có reward hacking hay không. Không có bằng chứng lần chạy này cải thiện suy luận; kết luận về chất lượng phải dựa vào tập kiểm tra. Cần lặp seed hoặc tăng tập test trước khi diễn giải thay đổi nhỏ này. GSM8K tiếng Anh ở NB6 khác dữ liệu GRPO tiếng Việt về ngôn ngữ và cách chấm, không nên so trực tiếp hai accuracy.

## Tình trạng bonus

- [x] NB3b: năm biến thể loss và phân tích độ dài.
- [x] NB5: GGUF Q4_K_M và so sánh HF/GGUF.
- [x] NB6: ba benchmark, stderr và diễn giải.
- [x] NB7: reward curve, accuracy trước/sau và nhiễu.
- [ ] β-sweep: chưa chạy ba giá trị β.
- [x] Chấm chéo API: OpenAI `gpt-4o-mini`, agreement 63,8% trên 58 prompt.
- [x] HF Hub: adapter DPO, tokenizer và model card công khai tại https://huggingface.co/long2711/lab22-dpo-vn-long-2A202602471. Adapter cần mô hình SFT đã gộp từ NB1 để tái tạo pipeline.

## Điều bất ngờ nhất

GRPO sau 60 bước giảm từ 51% xuống 49% trên 100 câu, trong khi DPO có margin reward dương nhưng win rate hội đồng chỉ 52%. Điều này nhắc tôi rằng reward tối ưu hóa và chất lượng đo trên tập kiểm tra là hai đại lượng khác nhau; báo cáo phải giữ cả kết quả chưa như kỳ vọng.
