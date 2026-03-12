import tensorflow as tf
import tf2onnx

model = tf.keras.models.load_model(r"../Saved_Models\BiLSTM_Baseline_final.keras")
input_signature = [tf.TensorSpec([None, 384, 708], tf.float32, name='input_features')]
tf2onnx.convert.from_keras(model, input_signature=input_signature, opset=13, output_path=r"../Saved_Models\BiLSTM_Baseline.onnx")
print("ONNX export done.")
