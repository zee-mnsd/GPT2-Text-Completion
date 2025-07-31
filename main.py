from GPT2TextCompletion import GPT2TextCompletion

if __name__ == '__main__':
    print("=== GPT-2 Text Completion Demo ===\n")

    # Khởi tạo hệ thống
    gpt2 = GPT2TextCompletion('gpt2')

    test_inputs = input("Nhập câu chưa hoàn chỉnh (nhập exit để thoát): ")

    while test_inputs != "exit":
        results = gpt2.complete_text(test_inputs)
        for result in results:
            print(f"Câu hoàn chỉnh là {result}")
        print("-" * 50)
        test_inputs = input("Nhập câu chưa hoàn chỉnh (nhập exit để thoát): ")
