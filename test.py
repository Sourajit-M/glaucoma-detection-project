import onnxruntime as ort

sess = ort.InferenceSession(
    r"D:\Machine Learning\glaucoma detection project\outputs\models\glaucoma_resnet18.onnx"
)

print("All intermediate node names:")
for node in sess.get_outputs():
    print(" output:", node.name, node.shape)

# Also check the model's internal nodes
import onnx
model = onnx.load(r"D:\Machine Learning\glaucoma detection project\outputs\models\glaucoma_resnet18.onnx")
print("\nLast 10 node output names:")
for node in model.graph.node[-10:]:
    print(" ", node.output)