import torch
from transformers import GPT2LMHeadModel, GPT2Tokenizer

class GPT2TextCompletion:
    def __init__(self, model_name='gpt2'):
        """ Khởi tạo mô hình GPT2

        Args:
            - model_name là tên model lấy từ OpenAI
                'gpt2', 'gpt2-medium', 'gpt2-large', 'gpt2-xl'

        Used tools:
            - tokenizer dùng để chuyển đổi văn bản thành token(computer hiểu)
            - model dùng để dự đoán tiếp theo
            - pad_token và eos_token để padding(đệm/lót thêm) và đánh dấu kết thúc cho token
        """
        # Thực hiện trên device nào? GPU(cuda) or CPU
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        print(f"Using device: {self.device}")

        # Tải tokenizer và model
        print(f"Loading {model_name} model...")
        self.tokenizer = GPT2Tokenizer.from_pretrained(model_name)
        self.model = GPT2LMHeadModel.from_pretrained(model_name)

        # Tinh chỉnh token về cùng độ dài
        self.tokenizer.pad_token = self.tokenizer.eos_token

        # Chuyển model lên GPU và bật eval - sử dụng model
        self.model.to(self.device)
        self.model.eval()

        print(f"{model_name} model loaded successfully!")

    def complete_text(self, input_text, max_new_tokens=50, temperature=0.8, top_k=50, top_p=0.9, num_return_sequences=1):
        """Hoàn thành văn bản

        Args:
            - input_text: văn bản đầu vào
            - max_new_tokens: số token mới được sinhh ra.
              sequence_length: Độ dài chuỗi token = max_new_tokens + input_length.
            - temperature: độ ngẫu nhiên(0.1-2.0)
              temperature càng cao thì sẽ chọn từ có xác suất xuất hiện thấp
              temperature càng thấp thì sẽ chọn từ có xác suất cao nhất
            - top_k và top_p đều dùng để lọc túi từ, chọn ra những từ có xác suất xuất hiện cao hơn
              top_k: chọn K token có xác suất cao nhất
              top_p: xem xét tokens có tổng xác suất <= top_p
            - num_return_sequences: số lượng kết quả trả về sau khi sinh.

        Biến và tham số cần quan tâm:
            - input_ids: là mã hoá input đầu vào thành các ID
            - return_tensors='pt' là trả về dạng tensor phù hợp Pytorch(pt) hoặc TensorFlow(tf)
            - do_sample: dùng để giảm việc chỉ chọn các từ có xác suất cao nhất
            - early_stopping: dừng sinh khi đạt điều kiện eos_token
            - no_repeat_ngram_size=2: tránh lặp lại cặp token đã có (tránh n-gram)
            - skip_special_tokens=True: bỏ qua các token đặc biệt
            - no_grad: vì không bật tính năng trainning nên không cần gradient

        Kết quả mong muốn:
            - outputs_ids: nhận những câu sinh ra (với num_return_sequences=1 thì sinh ra 1 câu)
            - results: chứa các câu hoàn chỉnh (trong trường hợp này là 1 câu)
        """
        # Mã hoá input_text
        input_ids = self.tokenizer.encode(input_text, return_tensors='pt').to(self.device)

        # Sinh văn bản
        with torch.no_grad():
            outputs_ids = self.model.generate(
                input_ids,
                max_new_tokens=max_new_tokens,
                temperature=temperature,
                top_k=top_k,
                top_p=top_p,
                num_return_sequences=num_return_sequences,
                pad_token_id=self.tokenizer.eos_token_id,
                do_sample=True,
                early_stopping=True,
                no_repeat_ngram_size=2
            )

        # Giải mã outputs_ids
        results = []
        for output in outputs_ids:
            completed_text = self.tokenizer.decode(output, skip_special_tokens=True)
            results.append(completed_text)

        return results
