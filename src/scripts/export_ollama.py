def convert_to_gguf(model_path):
    subprocess.run([
        "python3", "-m", "llama_cpp.convert",
        "--input", model_path,
        "--output", "ollama_model.gguf",
        "--quantize", "Q4_K_M"
    ])
