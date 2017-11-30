import os
import sys
import tensorflow as tf
def export(model, sess, signature_name, export_path, version):
    # export path
    export_path = os.path.join(export_path, signature_name, str(version))
    print('Exporting trained model to {} ...'.format(export_path))

    builder = tf.saved_model.builder.SavedModelBuilder(export_path)
    # Build the signature_def_map.
    input_data= tf.saved_model.utils.build_tensor_info(model.input_data)
    input_data_pin=tf.saved_model.utils.build_tensor_info(model.input_data_pin)
    targets=tf.saved_model.utils.build_tensor_info(model.targets)
    result=tf.saved_model.utils.build_tensor_info(model.predicting_logits)
    lr=tf.saved_model.utils.build_tensor_info(model.lr)
    target_sequence_length = tf.saved_model.utils.build_tensor_info(model.target_sequence_length)
    source_sequence_length = tf.saved_model.utils.build_tensor_info(model.source_sequence_length)

    prediction_signature = tf.saved_model.signature_def_utils.build_signature_def(
        inputs={'input_data': input_data, 'input_data_pin':input_data_pin,'targets':targets,
                'lr':lr,'target_sequence_length':target_sequence_length,'source_sequence_length':source_sequence_length},
        outputs={'output': result},
        method_name=tf.saved_model.signature_constants.PREDICT_METHOD_NAME)  # 'tensorflow/serving/predict'
    builder.add_meta_graph_and_variables(
        sess, [tf.saved_model.tag_constants.SERVING],
        signature_def_map={
            signature_name: prediction_signature
        })
    builder.save()
if __name__=='__main__':
    pass
