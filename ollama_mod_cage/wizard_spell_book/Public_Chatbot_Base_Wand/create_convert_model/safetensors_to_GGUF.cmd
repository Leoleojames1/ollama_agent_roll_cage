@echo on
cd %1
python llama.cpp\convert-hf-to-gguf.py --outtype q8_0 --model-name %2-q8_0 --outfile %1\converted\%2-q8_0.gguf %2
@REM python llama.cpp\convert-hf-to-gguf.py --outtype f16 --model-name %2-f16 --outfile %1\converted\%2-f16.gguf %2
@REM python llama.cpp\convert-hf-to-gguf.py --outtype f32 --model-name %2-f32 --outfile %1\converted\%2-f32.gguf %2