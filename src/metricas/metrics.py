import tensorflow as tf

def fuzzy_accuracy_tf(y_true, y_pred):
    """Versión TF compatible con Keras.metrics."""
    return tf.reduce_mean(
        tf.cast(tf.abs(tf.argmax(y_true,1)-tf.argmax(y_pred,1))<=1, tf.float32)
    )

def fpi(y_true, y_pred, weights=(1,0.5,0)):
    """
    Fuzzy Performance Index — penaliza distancias de clase:
      0 si idéntico, 
      0.5 si a una clase de distancia, 
      1 si >1.
    """
    d = tf.abs(tf.argmax(y_true,1)-tf.argmax(y_pred,1))
    w = tf.constant(weights, dtype=tf.float32)
    return tf.reduce_mean(tf.gather(w, d))
