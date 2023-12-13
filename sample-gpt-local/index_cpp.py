# from llama_cpp import Llama
# conda activate llama
# https://github.com/abetlen/llama-cpp-python
# https://huggingface.co/TheBloke/Mistral-7B-OpenOrca-GGUF
# "./models/mistral-7b-openorca.Q4_K_M.gguf"
from llama_cpp import Llama
llm = Llama(model_path="./models/mistral-7b-openorca.Q2_K.gguf",
            n_gpu_layers=1, n_ctx=4096)

output = llm.create_completion("""<|im_start|>system
You are a helpful chatbot.
<|im_end|>
<|im_start|>user
Hello, 1 + 2 = ? <|im_end|>
<|im_start|>assistant""", max_tokens=500,  stop=["<|im_end|>"], stream=True)

for token in output:
    print(token[ "choices"][0]["text"], end='', flush=True)