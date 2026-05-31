from graph_builder import DictGraphRAG

# 1. 实例化并构建
rag = DictGraphRAG(data_dir="data")
rag.build_graph()

# 2. 缓存下来，以后直接调用 rag.load_from_disk() 就不用重新读取 CSV 了
rag.save_to_disk("job_dict_graph.pkl")

# 3. 模拟 Agent 推理时的查询
print("\n--- 测试：调取节点特征 ---")
features = rag.get_job_features("2-05-01")
print("临床和口腔医师特征:", features)

print("\n--- 测试：触发致命红线 ---")
# 模拟向量检索同时找出了 2-05-01(医师) 和 4-01-01(销售)
wormhole_warning = rag.check_wormhole("2-05-01", "4-01-01")
print("红线警告:", wormhole_warning)

print("\n--- 测试：触发易混淆口诀 ---")
confuse_rule = rag.get_confusion_rule("5-05-99", "5-99-00")
print("大模型防坑口诀:", confuse_rule)