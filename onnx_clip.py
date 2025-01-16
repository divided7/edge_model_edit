import onnx
import argparse

def remove_nodes_from_model(model_path, output, node_names_to_remove):
    """
    从ONNX模型中删除指定的节点

    参数:
    - model_path: str，模型文件的路径
    - node_names_to_remove: list，包含需要删除的节点名称列表
    """
    # 加载ONNX模型
    model = onnx.load(model_path)

    # 获取模型的图 (graph)
    graph = model.graph
    print("graph:")
    print(type(graph))  # <class 'onnx.onnx_ml_pb2.GraphProto'>

    # 获取所有的节点（层）
    nodes = graph.node
    print("nodes: ")
    print("len(nodes): ", len(nodes))
    for node in nodes:
        # 每个node代表一个层，包含input：输入的名字，output：输出的名字，name：模型层的名字，op_type：这层是什么模型和属性
        print(node.input)
        print(node.output)
        print(node.name)
        print(node.op_type)
        print(node.attribute)

    # 检查每个待删除的节点名称是否存在
    missing_nodes = [node_name for node_name in node_names_to_remove if not any(node.name == node_name for node in nodes)]
    if missing_nodes:
        raise ValueError(f"以下节点不存在于模型中: {', '.join(missing_nodes)}")

    # 根据节点名称过滤出需要删除的节点
    nodes_to_remove = [node for node in nodes if node.name in node_names_to_remove]

    # 从图中移除这些节点
    for node in nodes_to_remove:
        graph.node.remove(node)

    # 检查节点是否已被删除
    print("更新后的节点列表:")
    for node in graph.node:
        print(node.name)

    # 保存修改后的模型
    onnx.save(model, output)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="删除ONNX模型中的指定节点")
    parser.add_argument("--input", type=str, default="input.onnx", help="原onnx文件名")
    parser.add_argument("--output", type=str, default="output.onnx", help="保存的onnx文件名")
    parser.add_argument("--node", type=str, nargs="+", help="需要删除的节点名称，使用空格分割")

    args = parser.parse_args()

    try:
        remove_nodes_from_model(args.input, args.output, args.node)
    except ValueError as e:
        print(f"错误: {e}")
