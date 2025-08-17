import os
import sys

def load_data(filename):
    """从文件中加载数据"""
    if not os.path.exists(filename):
        print(f"Error: {filename} not found!")
        return None
    
    with open(filename, 'r') as f:
        line = f.read().strip()
        data = list(map(float, line.split()))
    
    return data

def compare_data(ans_data, out_data, tolerance):
    """比较两个数据数组，返回不匹配的数据对"""
    if len(ans_data) != len(out_data):
        print(f"Error: Data length mismatch. ans_data: {len(ans_data)}, out_data: {len(out_data)}")
        return None
    
    mismatches = []
    for i, (ans, out) in enumerate(zip(ans_data, out_data)):
        if abs(ans - out) > tolerance:
            mismatches.append((ans, out, i))
    
    return mismatches

def compare_data_relative(ans_data, out_data, tolerance):
    """比较两个数据数组，返回不匹配的数据对"""
    if len(ans_data) != len(out_data):
        print(f"Error: Data length mismatch. ans_data: {len(ans_data)}, out_data: {len(out_data)}")
        return None
    
    mismatches = []
    for i, (ans, out) in enumerate(zip(ans_data, out_data)):
        if abs(ans - out)/abs(ans) > tolerance:
            mismatches.append((ans, out, i))
    
    return mismatches

def mian(fidx):
    # 加载数据
    ans_data = load_data(f"./out/{fidx}_pytorch.txt")
    out_data = load_data(f"./out/{fidx}_cpp.txt")
    # print("总数据个数：",len(out_data))
    
    if ans_data is None or out_data is None:
        return 1
    
    # print(f"Loaded {len(ans_data)} values from answer.test")
    # print(f"Loaded {len(out_data)} values from out.test")
    
    # 定义误差阈值
    # tolerances = [1e-1, 1e-2, 1e-3, 1e-4, 1e-5, 1e-6,1e-7]
    tolerances = [1e-1, 1e-2, 1e-3, 1e-4, 1e-5]
    
    for tolerance in tolerances:
        mismatches = compare_data(ans_data, out_data, tolerance)
        
        if mismatches is None:
            return 1
        
        if len(mismatches) == 0:
            # print(f"1e-{int(-round(math.log10(tolerance)))} 比较成功")
            continue
        else:
            print(f"IDX: {fidx}")
            print(f"1e-{int(-round(math.log10(tolerance)))} 比较失败，不匹配数据数量: {len(mismatches)}")
            
            # 输出前10个不匹配的数据
            # print("前10个不匹配的数据:")
            # for i, (ans, out, idx) in enumerate(mismatches[:10]):
            #     print(f"{ans:.6f}:{out:.6f} (index {idx})")
            
            # if len(mismatches) > 10:
            #     print(f"... 还有 {len(mismatches) - 10} 个不匹配的数据")

            print()
            return

if __name__ == "__main__":
    import math
    for i in range(0, 143):
        mian(i)