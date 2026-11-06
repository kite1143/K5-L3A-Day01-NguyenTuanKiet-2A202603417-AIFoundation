# Báo cáo cá nhân — Day 01: Lọc tin nhắn rác bằng ML, DL, Embedding và LLM

> Đổi tên file này thành `MSSV_HoTen.md` của bạn, ví dụ `2A20260000_NguyenVanAn.md`, rồi điền từng mục. Xoá các dòng hướng dẫn in nghiêng khi xong.

- **MSSV:2A202603417**
- **Họ và tên: Nguyễn Tuấn Kiệt**
- **SEED: 4480**

## 1. Bằng chứng

| Tệp | Có? |
| --- | --- |
| [`day01_lab.ipynb`](day01_lab.ipynb) — notebook đã chạy, còn output | v |
| [`lab1_report.json`](lab1_report.json) — do ô "Báo cáo cuối" sinh ra | v |
| [`hand_calc.jpg`](hand_calc.jpg) — ảnh tính tay có các bước | v |

## 2. Kết quả chính
                               precision  recall  F1 (test)  F1 (SUB 40)  \
1. BoW + LogReg (tự code)            1.000   0.835      0.910        0.857   
2. BoW + MLP (deep learning)         0.978   0.918      0.947        0.919   
3a. Encoder (MiniLM) + kNN           0.856   0.979      0.913        0.976   
3b. Encoder (MiniLM) + LogReg        0.940   0.969      0.954        0.976   
3c. Encoder (MiniLM) + MLP head      0.904   0.969      0.935        0.976   
4a. LLM nhỏ zero-shot                0.000   0.000      0.000        0.000   
4b. LLM lớn zero-shot                0.000   0.000      0.000        0.000   
4c. LLM nhỏ few-shot                 0.000   0.000      0.000        0.000   

                                 train (s)  tin/s (cả lô)  p50 ms/tin  \
1. BoW + LogReg (tự code)            6.999       8674.596       0.057   
2. BoW + MLP (deep learning)        12.539      22410.294       0.335   
3a. Encoder (MiniLM) + kNN          85.860         27.570      45.162   
3b. Encoder (MiniLM) + LogReg       86.285         23.836      27.762   
3c. Encoder (MiniLM) + MLP head     86.899         28.143      28.112   
4a. LLM nhỏ zero-shot                0.000          1.030     858.394   
4b. LLM lớn zero-shot                0.000          0.944    1044.270   
4c. LLM nhỏ few-shot                 0.000          0.695     846.086   

                                 p95 ms/tin  USD / 1k tin  
1. BoW + LogReg (tự code)             0.287         0.000  
2. BoW + MLP (deep learning)          0.486         0.000  
3a. Encoder (MiniLM) + kNN           58.851         0.000  
3b. Encoder (MiniLM) + LogReg        35.705         0.000  
3c. Encoder (MiniLM) + MLP head      38.953         0.000  
4a. LLM nhỏ zero-shot              1960.655         0.005  
4b. LLM lớn zero-shot              1202.184         0.049  
4c. LLM nhỏ few-shot               5956.656         0.016  
• Hàng LLM chỉ đo trên SUB (40 tin) → so sánh chất lượng công bằng bằng cột 'F1 (SUB 40)'.
• SUB chỉ 40 tin: 1 tin sai đổi F1 vài điểm. Chênh < ~0.05 F1 coi như HOÀ — đừng xếp hạng theo số lẻ.
• Inference đo TỪ VĂN BẢN THÔ → NHÃN (gồm cả vectorize / encode). p50/p95 = gọi từng tin một, như một request thật.
• train (s) của 3a/3b/3c = thời gian encoder MiniLM encode tập train (+ train đầu LogReg / MLP); LLM = 0 vì không train. Số đo phụ thuộc CPU/GPU của Colab.

## 3. Dự đoán và thực tế

1. Dự đoán llm_models (Cell 49 - Thí nghiệm 4.A)
   - Dự đoán: Model gpt-6.1-terra sẽ có F1 cao hơn gpt-6-luna, cả hai đạt F1 cao và chỉ chênh lệch khoảng 1–2%.
   - Kết quả thực tế: Cả 2 model đều đạt F1 = 0.000.
   - Vì sao lệch: Do cấu hình CLASSIFY_MAX_TOKENS = 3 kết hợp với cách hoạt động của mô hình. Khi token quá ngắn hoặc do chuỗi suy nghĩ/reasoning chiếm dụng hết token khiến output nhãn không sinh ra được hoặc không khớp với định dạng regex của parse_label, toàn bộ nhãn trả về None (fallback về 0), khiến cả Precision, Recall lẫn F1 về 0 tuyệt đối.

2. Dự đoán llm_fewshot (Cell 51 - Thí nghiệm 4.B)
   - Dự đoán: F1 sẽ tăng nhẹ so với zero-shot vì có thêm 4 ví dụ mẫu hướng dẫn định dạng.
   - Kết quả thực tế: F1 vẫn bằng 0.000 (invalid = 40).
   - Vì sao lệch: Thêm ví dụ vào prompt làm tăng độ dài input nhưng không giải quyết được rào cản ở đầu ra output token (max_tokens = 3 hoặc cấu hình reasoning). Khi output vẫn không xuất ra đúng nhãn spam/ham đơn lẻ cho parser đọc được, toàn bộ dự đoán tiếp tục bị gán nhãn 0, F1 không cải thiện mà giữ nguyên ở mức 0.

3. Dự đoán emb_search (Cell 32 - Thí nghiệm 3.A)
   - Dự đoán: Kết quả tìm kiếm gần nhất sẽ là ham vì cho rằng model chưa hiểu được ngữ cảnh.
   - Kết quả thực tế: Toàn bộ top kết quả tương đồng nhất trả về đều là spam (điểm tương đồng cao từ 0.513 đến 0.563).
   - Vì sao lệch: Câu truy vấn không chứa từ khóa trực tiếp như free, win, prize nhưng mang đầy đủ ngữ nghĩa của chiêu trò lừa đảo ("selected to receive a complimentary holiday, reply now to claim"). Không gian vector của mô hình embedding pre-train (all-MiniLM-L6-v2) hiểu được độ tương đồng ngữ nghĩa sâu sắc giữa "complimentary holiday" với "free award/prize", do đó gom chính xác các tin nhắn spam có cùng chủ đề lại gần nhau thay vì bắt nhầm sang ham.   

## 4. Phân tích 50 case khó
- Tụt nhiều nhất: BoW (MLP & LogReg), F1 rơi từ ~0.90 xuống ~0.33 (accuracy còn 0.50, ngang đoán mò).
- Lý do tập thường lạc quan là vì chỉ chứa spam rập khuôn; mô hình chỉ cần học vẹt vài từ khóa nổi bật (free, win) là đạt điểm cao mà không cần hiểu ngữ cảnh thực sự.
- 2 cases:
  - "FREE pizza in the office kitchen right now, come quick before it's gone": Chứa các từ Free, now dẫn đến bị đánh là spam
  - "C0ngr4tulat10ns! Y0u h4ve w0n a £1000 g1ft c4rd. Cl4im n0w at w1n-gift.c0m": Chứa các từ bị viết chệch đi bằng các số dẫn đến bị đánh là ham

## 5. Thử thách cá nhân
🎯 Đề #5: Embedding · Viết 4 tin nhắn của riêng bạn (2 spam né từ khoá, 2 ham chứa 'free'/'win'). So dự đoán BoW vs Embedding+LogReg và giải thích từng tin.
MY_CASES = [
    (
        "Y0ur p@ckage delivery is on h0ld. Verify y0ur address immediately at track-post.co",
        "spam",
        "obfuscation"
    ),
    (
        "Hi, this is Emily from HR. We reviewed your CV for a remote role ($200/day), add WhatsApp +44712345678 to start",
        "spam",
        "natural_spam"
    ),
    (
        "Are you free this evening? Mom made extra lasagna so drop by if you want",
        "ham",
        "ham_looks_spam"
    ),
    (
        "We must win the match tomorrow to stay in the tournament, see you at practice!",
        "ham",
        "ham_looks_spam"
    )
]
Câu: Y0ur p@ckage delivery is on h0ld. Verify y0ur address immediately at track-post.co
- Nhãn: Spam
- Dự đoán: BoW = Ham; Embedding + LogReg = Spam
- Giải thích:
  - BoW sai vì các từ như (Y0ur, p@ckage, h0ld) bị xem là từ ngoài từ điển của BoW nên nó dự đoán là ham.
  - Embedding đúng: Tokenizer tách nhỏ thành các subword; tầng Attention vẫn liên kết được ngữ cảnh.

Câu: Hi, this is Emily from HR. We reviewed your CV for a remote role ($200/day), add WhatsApp +44712345678 to start
- Nhãn: Spam
- Dự đoán: BoW = Ham; Embedding + LogReg = Spam
- Giải thích:
  - BoW sai: Toàn bộ từ ngữ đều lịch sự và không chứa từ khoá spam nên BoW đánh giá an toàn.
  - Embedding đúng: Nắm được ngữ cảnh của dạng lừa đảo việc làm (thu nhập ngày cao bất thường kết hợp điều hướng sang ứng dụng nhắn tin riêng tư)

Câu: Are you free this evening? Mom made extra lasagna so drop by if you want
- Nhãn: Ham
- Dự đoán: BoW = Spam; Embedding + LogReg = Ham
- Giải thích:
  - BoW sai: Từ 'free' có trọng số spam rất cao trong tập train; sự xuất hiện của nó làm điểm spam tăng vọt -> gây báo động giả.
  - Embedding đúng: Nhận diện cụm "Are you free" là câu hỏi rảnh rỗi trong giao tiếp bạn bè, kết hợp từ vựng sinh hoạt (Mom, lasagna, drop by) để xác định đúng nhãn Ham.

Câu: We must win the match tomorrow to stay in the tournament, see you at practice!
- Nhãn: Ham
- Dự đoán: BoW = Spam; Embedding + LogReg = Ham
- Giải thích:
  - BoW sai: Từ 'win' thường gắn liền với trúng thưởng trong dữ liệu spam, khiến mô hình đếm từ bắt nhầm thành spam.
  - Embedding đúng: Nắm bắt mối liên hệ ngữ nghĩa giữa 'win', 'match', 'tournament' và 'practice', hiểu đây là mục tiêu thi đấu thể thao chứ không phải thông báo nhận thưởng thụ động.

## 6. Phương án cho sản phẩm thật
Có 2 cách chọn model

* Cách 1: Chọn 1 model xử lý tất cả tin nhắn
Các cách đạt yêu cầu độ trễ (p95 < 50 ms):
- BoW + LogReg
- BoW + MLP
- Encoder (MiniLM) + LogReg
Cách đạt cả độ trễ lẫn chất lượng:
- Encoder (MiniLM) + LogReg
Ước tính chi phí/tháng nếu dùng LLM (300 triệu SMS/tháng)
- Với LLM nhỏ (gpt-6-luna, ~$0.005/1k tin): $10.5x10 = 105 USD/ngày -> ~3.150 USD/tháng.   
- Với LLM lớn (gpt-6.1-terra, ~$0.049/1k tin): $49x10 = 490 USD/ngày -> ~14.700 USD/tháng.

* cách 2:
Ta có thể tạo một hệ thống gồm 2 tầng
- Tầng 1 là Encoder (MiniLM) + LogReg vì vừa nhanh vừa nhẹ, cho 100% các tin nhắn chạy qua. Nếu tin nhắn nào có xác suất spam/ham mơ hồ thì chuyển sang tầng 2
- Tầng 2 là sử dụng LLM + prompt phòng thủ để đánh giá tin nhắn.

Lý do sử dụng hệ thống:
- Tối ưu hiệu suất trả lời nhanh và chính xác khi áp dụng cả 2 model.
- Tối ưu chi phí khi số lượng tin nhắn đưa vào LLM ít hơn hẳn.

## 7. AI log và điều đã học
### 🤖 AI log (bắt buộc nếu bạn dùng AI ở bất kỳ đâu)
| Phần | Bạn hỏi AI gì (prompt) | AI trả lời/viết gì | Bạn đã sửa / kiểm chứng gì |
| --- | --- | --- | --- |
| **Phần 2: Sentence Embedding & kNN** | Phát hiện 2 lỗi trong hàm `top_k_similar_v2` | Chỉ ra lỗi thiếu chuẩn hóa L2 (chỉ là dot product) và `np.argsort` xếp tăng dần khiến lấy nhầm top xa nhất | Tự kiểm tra lại công thức Cosine similarity, sửa lại code với `normalize` và đảo ngược mảng `[::-1][:k]`. |
| **Phần 4: LLM API (TODO 4.1 - 4.4)** | Nhờ viết code các hàm `call_openai`, `build_classification_prompt`, `parse_label`, `llm_classify_batch` | Cung cấp code gọi API có đo thời gian, regex bóc tách nhãn `spam`/`ham`, xử lý ngoại lệ và đếm `invalid` | Test chạy thực tế trên notebook; phát hiện lỗi `LocalProtocolError` do dính biến môi trường và tách lại `OPENAI_API_KEY`; kiểm chứng bộ bóc tách nhãn tránh bắt nhầm `not spam`. |
| **Phần 4: Phân tích & Bắt lỗi LLM** | Giải thích nguyên nhân $F_1 = 0$ ở 4.A/4.B, cơ chế `max_tokens`/`reasoning_effort`, và lỗi của hàm `parse_label_v2` | Giải thích do model suy luận cạn token trước khi sinh nội dung (`invalid=40`) và toán tử `in "spam"` bị dính bẫy phủ định `not spam` | Kiểm tra log kết quả thực tế trên notebook thấy đúng 40/40 mẫu trả về invalid; đối chiếu lý thuyết về token suy luận. |
| **Phần 5: Đánh đổi Pareto & SLA** | Dự đoán và phân tích các mô hình nằm trên đường Pareto và vi phạm SLA 50 ms | Phân tích `BoW + LogReg` (nhanh nhất), `Emb + LogReg` (tối ưu cân bằng), và các phương pháp LLM vượt xa SLA 50 ms | Đối chiếu trực tiếp với biểu đồ Pareto trực quan hóa từ notebook để xác nhận vị trí các điểm và đường biên tối ưu. |
| **Phần 6: Stress test & Tự tạo ca khó** | Phân tích 50 case khó, gợi ý 5 case đối kháng mới đánh lừa mô hình tốt nhất (`Emb + LogReg`) | Giải thích cơ chế thất bại theo cách nhìn văn bản của từng model; tạo 5 test case thuộc các nhóm `obfuscation`, `natural_spam`, `talks_about_spam` | Nạp trực tiếp `MY_CASES` vào DataFrame `HARD_EVAL`, chạy lại ô kiểm thử và xác nhận mô hình embedding bị lừa đúng như phân tích. |
