from onnxruntime.quantization import quantize_dynamic, QuantType

input_model = "artifacts/narratorflow-onnx/model.onnx"
output_model = "artifacts/narratorflow-onnx/model_quantized.onnx"

quantize_dynamic(
    model_input=input_model,
    model_output=output_model,
    weight_type=QuantType.QInt8,
    per_channel=True
)

print("INT8 quantization completed!")