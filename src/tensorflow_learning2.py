import tensorflow as tf
# 声明变量
state = tf.Variable(0,'first')
# 常量
one = tf.constant(1)
# 相加
new_state = tf.add(state,one)
# 更新变量
update = tf.assign(state,new_state)
# 初始化计算
init = tf.initialize_all_variables()
with tf.Session() as sess:
    sess.run(init)
    for i in range(3):
        sess.run(update)
        print(sess.run(state))

