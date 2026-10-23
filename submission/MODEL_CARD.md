---
language:
- vi
base_model: unsloth/Qwen3-4B-Instruct-2507-unsloth-bnb-4bit
tags:
- peft
- lora
- dpo
- vietnamese
---

# Day22 Vietnamese SFT + DPO adapter

**Tác giả:** Nguyễn Hải Long (2A202602471), Track 3.

Đây là **LoRA adapter DPO**, không phải mô hình đầy đủ. Adapter được huấn luyện trên mô hình SFT mini đã gộp từ `unsloth/Qwen3-4B-Instruct-2507-unsloth-bnb-4bit`. Muốn tái tạo chính xác, trước tiên chạy NB1 để tạo `models/sft-merged/`, sau đó tải adapter vào bản SFT đó. Chỉ riêng base model công khai không tái tạo được toàn bộ pipeline.

## Dữ liệu và phương pháp

- SFT: `saillab/alpaca-vietnamese-cleaned`, 1.000 mẫu, một epoch.
- Preference: `sailor2/sea-ultrafeedback-onpolicy`, 800 cặp train và 100 cặp held-out; tách theo prompt.
- DPO: β 0,1; learning rate 5e-6; một epoch; LoRA rank 16, alpha 32.
- Môi trường thực chạy: RTX 3060 12 GB, Python 3.11, cấu hình tier T4.

## Kết quả kiểm tra

- Reward accuracy held-out: 62%; reward gap held-out: +0,0414.
- Hội đồng hai reward model Skywork trên 50 prompt held-out: DPO win rate 52%, CI 95% [43%, 62%]. Khoảng này chứa 50%, vì vậy chưa chứng minh DPO tốt hơn SFT.
- IFEval: SFT 49,0%, DPO 50,0%; GSM8K: 82,0% → 85,2%; Global-MMLU-vi: 24,21% ở cả hai, với các giới hạn mẫu ghi trong `submission/REFLECTION.md`.
- GRPO là một thí nghiệm riêng, **không nằm trong adapter DPO này**; accuracy trước/sau 51% → 49% trên 100 câu.

## Giới hạn

Kết quả đánh giá dùng mẫu nhỏ và các reward model cùng nhóm phát triển với nguồn nhãn preference. Một số đầu ra giữ lại token `<tool_call>` không mong muốn; cần làm sạch template trước khi triển khai. Safety chỉ được kiểm tra trên bốn prompt cố định, nên adapter chưa thích hợp cho ứng dụng có yêu cầu an toàn cao.

Mã, notebook có output, biểu đồ và báo cáo: [GitHub Day22](https://github.com/long27112003/K4-L3-Track3-Day22-NguyenHaiLong-2A202602471).
