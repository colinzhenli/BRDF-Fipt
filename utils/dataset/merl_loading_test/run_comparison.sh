#!/bin/bash
# Script to compile C++ reference and compare with Python implementation

if [ "$#" -ne 1 ]; then
    echo "Usage: $0 <path_to_brdf.binary>"
    echo "Example: $0 /path/to/alum-bronze.binary"
    exit 1
fi

BRDF_FILE="$1"

if [ ! -f "$BRDF_FILE" ]; then
    echo "Error: BRDF file not found: $BRDF_FILE"
    exit 1
fi

echo "========================================"
echo "MERL BRDF Comparison Test"
echo "========================================"
echo "BRDF file: $BRDF_FILE"
echo ""

# Compile C++ reference implementation
echo "[1/4] Compiling C++ reference implementation..."
g++ -O3 -o brdf_read BRDFRead.cpp -lm
if [ $? -ne 0 ]; then
    echo "Error: Compilation failed"
    exit 1
fi
echo "✓ Compilation successful"
echo ""

# Run C++ implementation
echo "[2/4] Running C++ reference implementation..."
./brdf_read "$BRDF_FILE" > cpp_output.txt
if [ $? -ne 0 ]; then
    echo "Error: C++ execution failed"
    exit 1
fi
echo "✓ C++ output saved to cpp_output.txt"
echo "   Lines: $(wc -l < cpp_output.txt)"
echo ""

# Run Python implementation
echo "[3/4] Running Python implementation..."
cd ..
python merl_test.py "$BRDF_FILE" > merl_loading_test/python_output.txt
if [ $? -ne 0 ]; then
    echo "Error: Python execution failed"
    exit 1
fi
cd merl_loading_test
echo "✓ Python output saved to python_output.txt"
echo "   Lines: $(wc -l < python_output.txt)"
echo ""

# Compare outputs
echo "[4/4] Comparing outputs..."
echo ""

# Check if files have same number of lines
CPP_LINES=$(wc -l < cpp_output.txt)
PY_LINES=$(wc -l < python_output.txt)

if [ "$CPP_LINES" -ne "$PY_LINES" ]; then
    echo "✗ WARNING: Different number of output lines!"
    echo "  C++:    $CPP_LINES lines"
    echo "  Python: $PY_LINES lines"
    echo ""
fi

# Show first few lines of each
echo "First 5 lines from C++:"
head -5 cpp_output.txt
echo ""

echo "First 5 lines from Python:"
head -5 python_output.txt
echo ""

# Compute differences
echo "Computing differences..."
python3 << 'EOF'
import sys

def parse_rgb(line):
    parts = line.strip().split()
    return [float(x) for x in parts]

with open('cpp_output.txt') as f1, open('python_output.txt') as f2:
    cpp_lines = f1.readlines()
    py_lines = f2.readlines()
    
    max_diff = 0.0
    total_diff = 0.0
    count = 0
    errors = []
    
    for i, (cpp_line, py_line) in enumerate(zip(cpp_lines, py_lines)):
        try:
            cpp_rgb = parse_rgb(cpp_line)
            py_rgb = parse_rgb(py_line)
            
            for j in range(3):
                diff = abs(cpp_rgb[j] - py_rgb[j])
                max_diff = max(max_diff, diff)
                total_diff += diff
                count += 1
                
                if diff > 1e-5:
                    errors.append((i+1, j, cpp_rgb[j], py_rgb[j], diff))
        except:
            print(f"Parse error at line {i+1}")
    
    avg_diff = total_diff / count if count > 0 else 0
    
    print(f"Statistics:")
    print(f"  Total values compared: {count}")
    print(f"  Maximum difference:    {max_diff:.10f}")
    print(f"  Average difference:    {avg_diff:.10f}")
    print()
    
    if max_diff < 1e-5:
        print("✓ PERFECT MATCH! Implementations are identical within numerical precision.")
    elif max_diff < 1e-3:
        print("✓ EXCELLENT: Implementations match within 0.1% tolerance")
    elif max_diff < 0.01:
        print("✓ GOOD: Implementations match within 1% tolerance")
    else:
        print("⚠ WARNING: Significant differences detected")
        if len(errors) > 0:
            print(f"\nFirst 5 mismatches:")
            for line, ch, cpp_val, py_val, diff in errors[:5]:
                channels = ['R', 'G', 'B']
                print(f"  Line {line}, {channels[ch]}: C++={cpp_val:.6f}, Py={py_val:.6f}, diff={diff:.6f}")

EOF

echo ""
echo "========================================"
echo "Comparison complete!"
echo "========================================"
echo ""
echo "Output files:"
echo "  - cpp_output.txt: C++ reference output"
echo "  - python_output.txt: Python implementation output"


