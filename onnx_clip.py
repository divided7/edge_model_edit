import onnx
import argparse
import numpy as np
import onnxruntime as ort
from onnx import helper
from onnx import TensorProto

def update_output_node(graph, output_nodes_info):
    # 检查原本的输出节点
    print("-"*50)
    print("原本的输出节点:")
    for index,eachNode in enumerate(graph.output):
        print(f"{index}: {eachNode.name}")
        graph.output.remove(graph.output[index])
        print(f"已删除{eachNode.name}节点")
    print("-" * 50)


    # 确保模型仍然有有效的输出节点
    for node_info in output_nodes_info:
        node_name = node_info['node_name']
        output_name = node_info['output_name']
        output_shape = node_info['output_shape']
        output_type = node_info['output_type']

        # 查找节点
        node = next((n for n in graph.node if n.name == node_name), None)
        if not node:
            print(f"未找到节点: {node_name}")
            continue

        # 根据 output_type 创建对应的数据类型
        if output_type == 'float16':
            tensor_type = TensorProto.FLOAT16
        elif output_type == 'float32':
            tensor_type = TensorProto.FLOAT
        else:
            raise ValueError(f"不支持的数据类型: {output_type}")

        # 创建新的输出节点的ValueInfoProto
        output_node = helper.make_tensor_value_info(
            output_name,  # 输出节点名称
            tensor_type,   # 输出数据类型（根据提供的 output_type）
            output_shape   # 输出形状
        )

        # 将新的输出节点添加到图的输出列表中
        graph.output.append(output_node)
        print(f"已创建新的输出节点: {output_name}, 形状: {output_shape}, 数据类型: {tensor_type}")

def remove_nodes_from_model(model_path, output, node_names_to_remove, output_nodes_info):
    """
    从ONNX模型中删除指定的节点
    """
    # 加载ONNX模型
    model = onnx.load(model_path)
    graph = model.graph

    # 获取所有的节点（层）
    nodes = graph.node

    # 检查每个待删除的节点名称是否存在
    missing_nodes = [node_name for node_name in node_names_to_remove if not any(node.name == node_name for node in nodes)]
    if missing_nodes:
        raise ValueError(f"以下节点不存在于模型中: {', '.join(missing_nodes)}")

    # 根据节点名称过滤出需要删除的节点
    nodes_to_remove = [node for node in nodes if node.name in node_names_to_remove]

    # 从图中移除这些节点
    for node in nodes_to_remove:
        graph.node.remove(node)

    # 检查节点是否已被删除，并更新输出节点
    update_output_node(graph, output_nodes_info)

    # 保存修改后的模型
    onnx.save(model, output)
    print(f"修改后的模型已保存为 {output}")

def simulate_inference(model_path):
    """
    使用ONNX模型进行一次推理测试，模拟运行并打印结果。

    参数:
    - model_path: str，模型文件的路径
    """
    # 创建 ONNX Runtime 推理会话
    session = ort.InferenceSession(model_path)

    # 获取模型输入的名字及形状
    input_name = session.get_inputs()[0].name
    input_shape = session.get_inputs()[0].shape
    print(f"输入名: {input_name}, 输入形状: {input_shape}")

    # 生成随机输入数据（根据模型的输入形状）
    input_data = np.random.random(input_shape).astype(np.float32)

    # 执行推理
    outputs = session.run(None, {input_name: input_data})

    # 获取输出名称
    output_names = [output.name for output in session.get_outputs()]

    # 输出每个推理结果的名称和形状
    print("推理结果的名称和形状:")
    for i, output in enumerate(outputs):
        print(f"输出名: {output_names[i]}, 形状: {output.shape}")



if __name__ == "__main__":

    # ---------------------------------------- args ----------------------------------------
    input_model = 'rope_static_fp16.onnx'
    output_model = 'rope_static_fp16_clip.onnx'
    # 需要删除的层节点
    del_node = ['Reshape_243', 'Reshape_235', 'Concat_244', 'Transpose_245', '0_graph_output_output']
    # 需要增加的输出节点（默认删除原有的所有输出节点）
    add_output_node = [
        {'node_name': 'Concat_227', 'output_name': '761', 'output_shape': [1, 101, 17, 30], 'output_type': 'float16'},
        {'node_name': 'Concat_206', 'output_name': '733', 'output_shape': [1, 101, 34, 60], 'output_type': 'float16'}
    ]
    # 仿真推理验证前向传播
    simulate = True
    # ---------------------------------------- args ----------------------------------------

    try:
        # 删除节点操作
        remove_nodes_from_model(input_model, output_model, del_node, add_output_node)

        # 如果需要进行推理模拟，调用simulate_inference函数
        if simulate:
            print("开始推理模拟...")
            simulate_inference(output_model)

    except ValueError as e:
        print(f"错误: {e}")
