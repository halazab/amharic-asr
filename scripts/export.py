"""
Export trained model to ONNX with quantization.
"""

import os
import torch
import numpy as np
from pathlib import Path
from typing import Optional


def export_to_onnx(
    model_path: str,
    output_path: str,
    opset_version: int = 17,
    dynamic_axes: bool = True
):
    """Export PyTorch model to ONNX."""
    print(f"Exporting model to ONNX: {output_path}")
    
    # Load model
    checkpoint = torch.load(model_path, map_location="cpu")
    
    # Create model architecture
    from scripts.train import ModelArchitecture
    arch = ModelArchitecture(vocab_size=1024)
    model = arch.create_model()
    model.load_state_dict(checkpoint["model_state_dict"])
    model.eval()
    
    # Create dummy input
    dummy_audio = torch.randn(1, 1, 16000)  # 1 second at 16kHz
    
    # Export
    output_dir = Path(output_path).parent
    output_dir.mkdir(parents=True, exist_ok=True)
    
    torch.onnx.export(
        model,
        dummy_audio,
        output_path,
        opset_version=opset_version,
        input_names=["audio"],
        output_names=["logits"],
        dynamic_axes={
            "audio": {0: "batch_size", 2: "audio_length"},
            "logits": {0: "batch_size", 1: "sequence_length"}
        } if dynamic_axes else None
    )
    
    print(f"ONNX model saved to: {output_path}")
    
    return output_path


def quantize_int8(
    onnx_path: str,
    output_path: str,
    per_channel: bool = True
):
    """Quantize ONNX model to INT8."""
    from onnxruntime.quantization import quantize_dynamic, QuantType
    
    print(f"Quantizing to INT8: {output_path}")
    
    # Quantize
    quantize_dynamic(
        onnx_path,
        output_path,
        weight_type=QuantType.QInt8
    )
    
    # Get file sizes
    original_size = os.path.getsize(onnx_path) / (1024 * 1024)
    quantized_size = os.path.getsize(output_path) / (1024 * 1024)
    
    print(f"INT8 quantization complete:")
    print(f"  Original: {original_size:.2f} MB")
    print(f"  Quantized: {quantized_size:.2f} MB")
    print(f"  Compression: {original_size / quantized_size:.2f}x")
    
    return output_path


def quantize_int4(
    onnx_path: str,
    output_path: str,
    block_size: int = 32
):
    """Quantize ONNX model to INT4 (k-quant style)."""
    from onnxruntime.quantization import quantize_dynamic, QuantType
    
    print(f"Quantizing to INT4: {output_path}")
    
    # Note: INT4 quantization may not be directly available in onnxruntime
    # This is a simplified version - real implementation would use k-quant
    
    # Fallback to INT8
    print("Warning: INT4 not directly supported, using INT8 instead")
    return quantize_int8(onnx_path, output_path)


def optimize_onnx(
    onnx_path: str,
    output_path: str,
    optimization_level: str = "all"
):
    """Optimize ONNX model with graph optimizations."""
    from onnxruntime.transformers import optimizer
    
    print(f"Optimizing ONNX model: {output_path}")
    
    # Optimize
    optimized_model = optimizer.optimize_model(
        onnx_path,
        model_type="bert",  # Generic transformer optimization
        num_heads=4,
        hidden_size=256
    )
    
    # Save
    optimized_model.save_model_to_file(output_path)
    
    print(f"Optimized model saved to: {output_path}")
    
    return output_path


def export_tensorrt(
    onnx_path: str,
    output_path: str,
    max_batch_size: int = 16,
    fp16: bool = True
):
    """Export to TensorRT engine."""
    print(f"Exporting to TensorRT: {output_path}")
    
    # Check if trtexec is available
    import subprocess
    
    cmd = [
        "trtexec",
        f"--onnx={onnx_path}",
        f"--saveEngine={output_path}",
        f"--maxBatch={max_batch_size}",
        "--workspace=1024"
    ]
    
    if fp16:
        cmd.append("--fp16")
    
    try:
        subprocess.run(cmd, check=True, capture_output=True)
        print(f"TensorRT engine saved to: {output_path}")
    except subprocess.CalledProcessError as e:
        print(f"TensorRT export failed: {e}")
        print("TensorRT SDK may not be installed")
        return None
    
    return output_path


def validate_onnx(onnx_path: str):
    """Validate ONNX model."""
    import onnxruntime as ort
    
    print(f"Validating ONNX model: {onnx_path}")
    
    try:
        session = ort.InferenceSession(onnx_path)
        
        # Check inputs/outputs
        inputs = session.get_inputs()
        outputs = session.get_outputs()
        
        print(f"Inputs: {[inp.name for inp in inputs]}")
        print(f"Outputs: {[out.name for out in outputs]}")
        
        # Run test inference
        dummy_input = np.random.randn(1, 1, 16000).astype(np.float32)
        result = session.run(None, {"audio": dummy_input})
        
        print(f"Test inference successful")
        print(f"Output shape: {result[0].shape}")
        
        return True
        
    except Exception as e:
        print(f"Validation failed: {e}")
        return False


def get_model_info(onnx_path: str):
    """Get ONNX model information."""
    import onnxruntime as ort
    
    session = ort.InferenceSession(onnx_path)
    
    # Get model info
    inputs = session.get_inputs()
    outputs = session.get_outputs()
    
    # Calculate parameter count (approximate)
    param_count = 0
    for input in inputs:
        shape = input.shape
        if isinstance(shape, list):
            param_count += np.prod([s for s in shape if isinstance(s, int)])
    
    info = {
        "path": onnx_path,
        "size_mb": os.path.getsize(onnx_path) / (1024 * 1024),
        "inputs": [{"name": inp.name, "shape": inp.shape, "type": inp.type} for inp in inputs],
        "outputs": [{"name": out.name, "shape": out.shape, "type": out.type} for out in outputs],
        "provider": session.get_providers()[0]
    }
    
    return info


def main():
    """Main export entry point."""
    import argparse
    
    parser = argparse.ArgumentParser(description="Export Amharic ASR model")
    parser.add_argument("--model", type=str, required=True, help="Path to PyTorch model")
    parser.add_argument("--output-dir", type=str, default="models", help="Output directory")
    parser.add_argument("--export-onnx", action="store_true", help="Export to ONNX")
    parser.add_argument("--quantize-int8", action="store_true", help="Quantize to INT8")
    parser.add_argument("--quantize-int4", action="store_true", help="Quantize to INT4")
    parser.add_argument("--optimize", action="store_true", help="Optimize ONNX model")
    parser.add_argument("--tensorrt", action="store_true", help="Export to TensorRT")
    parser.add_argument("--validate", action="store_true", help="Validate exported model")
    
    args = parser.parse_args()
    
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Export ONNX
    if args.export_onnx:
        onnx_path = str(output_dir / "amharic_asr.onnx")
        export_to_onnx(args.model, onnx_path)
    
    # Quantize
    onnx_path = str(output_dir / "amharic_asr.onnx")
    
    if args.quantize_int8:
        int8_path = str(output_dir / "amharic_asr_int8.onnx")
        quantize_int8(onnx_path, int8_path)
    
    if args.quantize_int4:
        int4_path = str(output_dir / "amharic_asr_int4.onnx")
        quantize_int4(onnx_path, int4_path)
    
    # Optimize
    if args.optimize:
        optimized_path = str(output_dir / "amharic_asr_optimized.onnx")
        optimize_onnx(onnx_path, optimized_path)
    
    # TensorRT
    if args.tensorrt:
        trt_path = str(output_dir / "amharic_asr_trt.plan")
        export_tensorrt(onnx_path, trt_path)
    
    # Validate
    if args.validate:
        for model_file in output_dir.glob("*.onnx"):
            validate_onnx(str(model_file))


if __name__ == "__main__":
    main()
