#如和使用循环神经网络（RNN）模型训练自己的问答小语言模型
#
#2023/5/25 13:56:58 复制
#要训练自己的问答小语言模型，首先需要准备一个合适的数据集。可以从互联网上寻找相应主题的QA数据集，或者手动创建一个数据集来训练模型。
#
#接下来，对于循环神经网络RNN的模型，需要使用Python编程语言及其相关库来实现。常用的库包括TensorFlow、Keras等，它们提供了方便的API和工具来搭建和训练模型。
#
#下面是一个基本的RNN模型训练代码示例：

# 导入库
import pandas as pd
import tensorflow as tf
from tensorflow.keras.layers import LSTM, Dense
from tensorflow.keras.models import Sequential

from tensorflow.keras.models import load_model
import numpy as np
#df=pd.read_excel("问答数据.xlsx",sheet_name="问答")
# 准备数据集
#data = ["question1", "answer1", "question2", "answer2", ...]
# 处理数据集
#questions = list(df['问题'])
#answers = list(df['答案'])
questions =['中国移动CDN系统整体架构是什么？',
 '江苏移动CDN系统介绍？',
 '中国移动CDN优势',
 '移动CDN商用怎么收费',
 '商用前需要客户提供什么？',
 'CDN分发服务实施流程是什么？',
 '业务洽谈主要谈哪些内容？',
 '需求评估涉及哪些内容？',
 '哪些网站不建议做CDN分发？',
 '哪些网站可以做CDN分发？',
 '怎么分析网站域名？',
 '静态资源类型有哪些？',
 '需求评估后输出什么？',
 '需求评估后注意事项？',
 '相关材料模板有哪些？',
 'CDN分发技术方案实施步骤？',
 '如何做好CDN分发后的质量保障？',
 '商用前需要客户提供什么？',
 '商用前期主要与客户沟通哪些事项？']
['中国移动CDN系统整体架构是什么？',
 '江苏移动CDN系统介绍？\n',
 '中国移动CDN优势\n',
 '移动CDN商用怎么收费',
 '商用前需要客户提供什么？',
 'CDN分发服务实施流程是什么？',
 '业务洽谈主要谈哪些内容？',
 '需求评估涉及哪些内容？',
 '哪些网站不建议做CDN分发？',
 '哪些网站可以做CDN分发？',
 '怎么分析网站域名？',
 '静态资源类型有哪些？',
 '需求评估后输出什么？',
 '需求评估后注意事项？',
 '相关材料模板有哪些？',
 'CDN分发技术方案实施步骤？',
 '如何做好CDN分发后的质量保障？',
 '商用前需要客户提供什么？',
 '商用前期主要与客户沟通哪些事项？']
answers = ['中国移动内容网络包括三层：内容管理层（北京）、调度分发层（CDN控制调度中心（北京）、CDN内容中心（北上广+成都+南京/无锡）)、边缘服务层（31省）。边缘节点全网31个省级节点、289个地市节点的部署，容量达到43.5Tbps ，分发总流量19.6Tbps',
 '1、江苏移动CDN节点容量10750Gbps。分发总流量峰值4564Gbps。\n2、部署南京和无锡2个省级节点，13个地市节点。\n3、已承载腾讯视频、芒果、乐视、爱奇艺、PPTV、凤凰等业务。\n',
 '1、覆盖全网，真正无死角\n2、调度精确，一步到位\n3、带宽充足，服务无瓶颈\n4、安全透明，全面防护\n',
 '请联系客户经理进行具体资费了解。',
 '信息安全责任书、CDN分发需求调研表、CDN客户基础信息收集表、HTTPS网站列表，证书或者证书获取方式、网站引入授权书',
 '业务洽谈（工作内容：与客户进行洽谈，信息交换，签署商务合同、安全协议；责任单位：分公司政企部门、分公司网络部门）\n需求评估（工作内容：依据客户需求与现网架构，评估是否可加速并确定加速方案；责任单位：分公司网络部门、省公司网络部）\n方案实施（工作内容：业务适配、定制开发、业务割接测试；责任单位：分公司网络部门、省公司网络部）\n质量保障（工作内容：保证分发质量；责任单位：分公司网络部门）',
 '服务内容 （简单的网站服务信息：网站域名、IP地址 )\n服务资费（商用收费）\n安全责任（主要用于确保网站内容的合法性）',
 '服务内容（根据收集的ICP客户需求加速内容、域名、端口、资源后缀等信息，进行加速内容服务的可行性分析。）\n调度方式（根据ICP客户侧要求的加速区域、客户的技术水平，如要求全国性  加速、拥有智能解析DNS系统等，来确认最终加速服务的调度方式）\n回源方式（根据ICP客户方的要求情况，明确通过回源方式获取还是ICP源站注  入方式）\n定制开发（根据ICP客户针对加速服务的一些特殊要求，如计费/定制报表等，  进行定制开发的工作量评估。）',
 '1、网站不能正常访问；\n2、网站首页为专有的注册、登录、搜索等页面；\n3、证券、银行、保险等金融公司的核心业务（金融公司的首页以图片或者控件、软件下载为    主可以推荐CDN分发）；\n4、文本信息为主的BBS论坛类网站；\n5、易受攻击的网站，比如医药、博彩类。',
 '1、于静态内容单独部署的域名，较容易进行分发；\n2、对于动态、静态内容混合的域名，当静态内容超过70%的，建议分发；\n3、对于动态内容占比较高的域名，由于代理回源流量较大，不建议分发',
 '可以借助互联网内容分析云iCAT-快猫，可以对待分发网站使用iCAT进行网站元素爬取，粗略计算静态内容占比',
 '图片（bmp|jpg|jpeg|png|tiff|gif|ico|svg|psd|cdr|pcd）\n网页（js|css|swf|txt|woff|ttf）\n下载（rar|zip|gzip|pdf|apk|gz|tgz|7z|exe|doc|xlsx|ppt|xls|jar|tar|docx|ipa）\n音/视频（mp4|flv|mov|wmv|rmvb|f4v|ts|m3u8|mkv|avi|3gp|mpg|mpeg|asf|rm|flash|mid|wav|wma|ogg|mp2|mka|mp3）\n\n\n',
 '1、对于零星网站：地市网络部门协同客户经理与客户进行交流，收集分发内容、域名、端口等信息，协助客户完成《中国移动CDN分发需求调研表》填写。（对于客户无法准确提供的，请地市公司网络部门通过抓包、拨测等手段辅助完成。\n2、对于代理商批量网站：协助客户完成《中国移动CDN分发需求调研表（XX代理XX个业务汇总）》和《网站引入授权书》\n',
 '1、确认待分发网站是否是http、https、http和https混合，如果包含https，请客户提供证书。\n2、如果客户已经采用其他CDN进行业务分发，请客户提供真正的源站IP，而非其他CDN地址。\n3、确认客户的权威DNS上是否可以区分中国移动地址进行cname配置，请参考《中国移动CDN引入客户指导(江苏公司)》中操作。\n',
 '详见链接http://www.baidu.com',
 '1、定商务细节后，与客户沟通上线时间，由省公司网络部牵头完成业务上线：资料审核-数据制作-业务上线\n1.1 资料审核：核实域名源地址，网站打开情况，域名正确性；部署拨测任务\n1.2 数据制作：杭研提交业务分发工单给集团，同步提交软件适配需求给杭研；华为提交软件适配需求给华为，华为反馈规则库，提交业务分发工单给集团；测试集团调度中心解析情况（判断集团数据制作是否正确）；在PC机上修改host文件测试规则库（判断规则库开发是否正确）\n1.3 业务上线：根据与客户沟通的上线时间，割接上线：客户在权威DNS上部署cname/省公司在local DNS上做DNS Forward；地市网络部门及时测试实际分发业务情况，及时与客户互动反馈情况\n2、分发效果跟踪：在确定提交业务分发需求之前，将待分发网站加入飞思达拨测系统，业务上线后对网站分发前后的效果进行对比（时延、丢包率等指标）',
 '1、运营报表：报表系统：https://cdn.4ggogo.com，以客户手机或邮箱为账号，创建业务运营报表帐号。地市网络部门负责指导客户使用\n2、运营跟踪：地市政企和地市网络加强与客户的互动，及时跟踪客户对于业务分发的反馈，有问题及时沟通解决。如果客户源站回源信息变更，发送邮件并附上《源站变更申请模板》给杭研徐烈或徐娜',
 '信息安全责任书、CDN分发需求调研表、CDN客户基础信息收集表、HTTPS网站列表，证书或者证书获取方式',
 '1、客户权威DNS是否支持智能DNS？\n2、客户网站是否已采用其他CDN进行分发？\n3、客户网站是否是https类型？\n4、其他需要沟通的事宜：\n1）网站全站或部分内容更新时间是否有要求？\n2）网站是否有防盗链要求？\n3）网站分发区域要求？\n4）其他事宜。\n\n']
#构建一个word to index的字典来将问题和答案转化为整数序列：
tokenizer = tf.keras.preprocessing.text.Tokenizer(filters='')
tokenizer.fit_on_texts(questions + answers)
question_seqs = tokenizer.texts_to_sequences(questions)
answer_seqs = tokenizer.texts_to_sequences(answers)

#对X和Y进行填充（padding），使得它们的长度相等。
max_length = max([len(a) for a in answer_seqs])
padded_answer_seqs = tf.keras.preprocessing.sequence.pad_sequences(answer_seqs, maxlen=max_length+1, padding='post')

max_length = max([len(q) for q in question_seqs])
padded_question_seqs = tf.keras.preprocessing.sequence.pad_sequences(question_seqs, maxlen=max_length+1, padding='post')



# 搭建模型
model = Sequential()
model.add(LSTM(128, input_shape=(1,2)))
model.add(Dense(len(answers)))
model = Sequential()
model.add(LSTM(128))
model.add(Dense(len(tokenizer.word_index) + 1, activation='softmax'))
# 编译模型
num_epochs = 10
model.compile(loss='sparse_categorical_crossentropy', optimizer='adam', metrics=['accuracy'])


#model.compile(loss='categorical_crossentropy', optimizer='adam')
#
## 训练模型
#for i in range(num_epochs):
#    for question, answer in zip(questions, answers):
#        X = [[vocab[c] for c in question]]
#        Y = [vocab[c] for c in answer]
#        model.train_on_batch(X, Y)
#
## 测试模型

batch_size = 2   # 每个训练批次的样本数，可以根据您的内存和 GPU 大小进行修改

# 使用 np.reshape() 将输入序列和答案序列转换为 3D 张量
padded_question_seqs = np.reshape(padded_question_seqs, (len(padded_question_seqs), max_length, 1))

padded_question_seqs = np.reshape(padded_question_seqs, (len(padded_question_seqs), max_length, 1))


padded_answer_seqs = np.reshape(padded_answer_seqs, (len(padded_answer_seqs), max_length, 1))

# 打印转换后的张量形状以确保正确
print('padded_question_seqs shape:', padded_question_seqs.shape)
print('padded_answer_seqs shape:', padded_answer_seqs.shape)

# 在 fit() 中指定 batch_size 参数
model.fit(padded_question_seqs, tf.keras.utils.to_categorical(padded_answer_seqs, num_classes=len(tokenizer.word_index) + 1), 
          epochs=num_epochs, batch_size=batch_size)

#model.fit(padded_question_seqs, tf.keras.utils.to_categorical(padded_answer_seqs, num_classes=len(tokenizer.word_index) + 1), epochs=num_epochs)
model.save('my_model.h5')

def predict_answer(question):
    X = [[vocab[c] for c in question]]
    yhat = model.predict(X)
    return ''.join([inv_vocab[np.argmax(y)] for y in yhat])
model.save('my_model.h5')
model = load_model('my_model.h5')

yhat = model.predict("你好")
print(''.join([inv_vocab[np.argmax(y)] for y in yhat]))
while True:
    question=input("请输入你的问题")
    if question=="break":
        break
    else:
        yhat = model.predict(X)
        print(''.join([inv_vocab[np.argmax(y)] for y in yhat]))
    
#在这个示例中，使用了LSTM层和Dense层来搭建模型，并使用categorical_crossentropy作为损失函数、adam作为优化器进行训练。训练完成后，可以通过predict_answer函数对模型进行测试。
#
#需要注意的是，在实际训练和测试过程中，需要进行数据预处理和清洗、使用
