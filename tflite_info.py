import tensorflow as tf
import numpy as np
import os


def analyze_tflite_model(model_path):
    # Load the model
    interpreter = tf.lite.Interpreter(model_path=model_path)
    interpreter.allocate_tensors()

    # Get model details
    input_details = interpreter.get_input_details()
    output_details = interpreter.get_output_details()

    # Get tensor details using the public API
    tensor_details = interpreter.get_tensor_details()

    # Print model overview
    print("\n" + "=" * 50)
    print(f"MODEL STRUCTURE: {os.path.basename(model_path)}")
    print("=" * 50)

    # Input information
    print("\nINPUT TENSORS:")
    for i, input_detail in enumerate(input_details):
        print(f"  Input #{i + 1}:")
        print(f"    Name: {input_detail['name']}")
        print(f"    Shape: {input_detail['shape']}")
        print(f"    Type: {input_detail['dtype']}")
        if 'quantization_parameters' in input_detail and len(input_detail['quantization_parameters']['scales']) > 0:
            print(
                f"    Quantization: scale={input_detail['quantization_parameters']['scales'][0]}, "
                f"zero_point={input_detail['quantization_parameters']['zero_points'][0]}")

    # Output information
    print("\nOUTPUT TENSORS:")
    for i, output_detail in enumerate(output_details):
        print(f"  Output #{i + 1}:")
        print(f"    Name: {output_detail['name']}")
        print(f"    Shape: {output_detail['shape']}")
        print(f"    Type: {output_detail['dtype']}")
        if 'quantization_parameters' in output_detail and len(output_detail['quantization_parameters']['scales']) > 0:
            print(
                f"    Quantization: scale={output_detail['quantization_parameters']['scales'][0]}, "
                f"zero_point={output_detail['quantization_parameters']['zero_points'][0]}")

    # Tensor information
    print("\nALL MODEL TENSORS:")
    for i, tensor in enumerate(tensor_details):
        print(f"  Tensor #{i} (index: {tensor['index']}):")
        print(f"    Name: {tensor['name']}")
        print(f"    Shape: {tensor['shape']}")
        print(f"    Type: {tensor['dtype']}")
        if 'quantization_parameters' in tensor and len(tensor['quantization_parameters']['scales']) > 0:
            print(
                f"    Quantization: scale={tensor['quantization_parameters']['scales'][0]}, "
                f"zero_point={tensor['quantization_parameters']['zero_points'][0]}")

    # Try to analyze the model structure using a more compatible approach
    try:
        # Load the flatbuffer model
        with open(model_path, 'rb') as f:
            model_buffer = f.read()

        # Use TFLite's native API to get model details
        model = tf.lite.experimental.Analyzer.analyze_model(model_buffer)

        if model and 'subgraphs' in model and model['subgraphs']:
            print("\nMODEL OPERATIONS (LAYERS):")
            op_types = {}

            for subgraph in model['subgraphs']:
                if 'operators' in subgraph:
                    for i, op in enumerate(subgraph['operators']):
                        op_name = op.get('opcode_name', 'UNKNOWN')

                        print(f"  Operation #{i}:")
                        print(f"    Type: {op_name}")

                        # Count operation types
                        op_types[op_name] = op_types.get(op_name, 0) + 1

                        # Show inputs
                        print(f"    Inputs:")
                        for j, input_idx in enumerate(op.get('inputs', [])):
                            if input_idx >= 0 and input_idx < len(tensor_details):
                                tensor = tensor_details[input_idx]
                                print(f"      - Tensor #{input_idx}: {tensor['name']} {tensor['shape']}")
                            else:
                                print(f"      - Tensor #{input_idx}: <unknown>")

                        # Show outputs
                        print(f"    Outputs:")
                        for j, output_idx in enumerate(op.get('outputs', [])):
                            if output_idx >= 0 and output_idx < len(tensor_details):
                                tensor = tensor_details[output_idx]
                                print(f"      - Tensor #{output_idx}: {tensor['name']} {tensor['shape']}")
                            else:
                                print(f"      - Tensor #{output_idx}: <unknown>")

            # Print summary of layer types
            print("\nLAYER TYPES:")
            for op_type, count in sorted(op_types.items()):
                print(f"  {op_type}: {count}")
    except Exception as e:
        print(f"\nCould not extract detailed operation information: {str(e)}")
        print("Showing basic structure only")

        # Alternative approach - try to infer operations from tensor names
        print("\nINFERRED OPERATIONS (based on tensor names):")
        op_types = {}

        # Group tensors by potential operations based on naming patterns
        operation_groups = {}
        for tensor in tensor_details:
            name = tensor['name']
            # Try to extract operation name from tensor name (common naming patterns)
            parts = name.split('/')
            if len(parts) > 1:
                op_name = parts[0]
                if op_name not in operation_groups:
                    operation_groups[op_name] = []
                operation_groups[op_name].append(tensor)

        # Print the inferred operations
        for i, (op_name, tensors) in enumerate(operation_groups.items()):
            # Try to guess the operation type from the name
            op_type = "Unknown"
            if "conv" in op_name.lower():
                op_type = "CONV"
            elif "dense" in op_name.lower() or "fc" in op_name.lower():
                op_type = "FULLY_CONNECTED"
            elif "pool" in op_name.lower():
                op_type = "POOL"
            elif "batch" in op_name.lower() and "norm" in op_name.lower():
                op_type = "BATCH_NORMALIZATION"
            elif "add" in op_name.lower():
                op_type = "ADD"
            elif "concat" in op_name.lower():
                op_type = "CONCATENATION"
            elif "softmax" in op_name.lower():
                op_type = "SOFTMAX"

            print(f"  Operation #{i}:")
            print(f"    Name: {op_name}")
            print(f"    Inferred Type: {op_type}")
            print(f"    Associated Tensors: {len(tensors)}")

            # Count operation types
            op_types[op_type] = op_types.get(op_type, 0) + 1

        # Print summary of inferred layer types
        print("\nINFERRED LAYER TYPES:")
        for op_type, count in sorted(op_types.items()):
            if op_type != "Unknown":
                print(f"  {op_type}: {count}")

    # Try to extract metadata if available
    try:
        metadata = interpreter.get_metadata_buffer()
        if metadata is not None:
            print("\nMODEL METADATA:")
            print(f"  Metadata buffer size: {len(metadata)} bytes")
    except Exception as e:
        print(f"\nCould not extract metadata: {str(e)}")

    # Summary statistics
    print("\nMODEL SUMMARY:")
    print(f"  Total tensors: {len(tensor_details)}")
    print(f"  Input tensors: {len(input_details)}")
    print(f"  Output tensors: {len(output_details)}")

    # Calculate approximate model size
    model_size_bytes = os.path.getsize(model_path)
    print(f"  Model file size: {model_size_bytes / (1024 * 1024):.2f} MB")


# Usage
if __name__ == "__main__":
    import sys

    if len(sys.argv) > 1:
        model_path = sys.argv[1]
    else:
        model_path = "bn/BirdNET_GLOBAL_6K_V2.4_Model_FP32.tflite"

    analyze_tflite_model(model_path)